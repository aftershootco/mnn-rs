//! Compare inference results between two backends on the same model.
//!
//! Fills every input with a deterministic pseudo-random pattern, runs the
//! model on a reference backend and a candidate backend, and reports
//! per-output element-wise differences:
//!
//! ```sh
//! cargo run -F cuda --release --example parity -- \
//!     examples/assets/paddle_rec_v4.mnn --candidate cuda --shape 1,3,48,320
//! ```
use mnn::*;
use std::path::PathBuf;

#[derive(Debug, clap::Parser)]
struct Cli {
    /// Path to the MNN model
    model: PathBuf,

    /// Reference backend
    #[arg(long, default_value = "cpu")]
    reference: ForwardType,

    /// Candidate backend to verify against the reference
    #[arg(short, long)]
    candidate: ForwardType,

    /// Resize all inputs to this shape before running (e.g. 1,3,48,320);
    /// models with dynamic dims need one.
    #[arg(short, long, value_delimiter = ',')]
    shape: Option<Vec<i32>>,

    /// Absolute tolerance for counting an element as mismatched
    #[arg(long, default_value = "1e-3")]
    tolerance: f32,

    /// Capture every operator's first output via session callbacks and report
    /// the ops whose outputs diverge (in reference execution order), instead
    /// of only comparing the model outputs.
    #[arg(long)]
    per_op: bool,

    /// Fill inputs from this file of raw little-endian f32s instead of the
    /// synthetic pattern (length must match the input element count)
    #[arg(long)]
    input_file: Option<PathBuf>,

    /// With --per-op: also write every matched op's stats (execution order)
    /// to this file as tab-separated lines
    #[arg(long)]
    dump_stats: Option<PathBuf>,

    /// Precision mode for the candidate backend
    #[arg(long, default_value = "high")]
    precision: PrecisionMode,

    /// With --per-op: dump both backends' values for the first op whose name
    /// contains this substring (as <dump>.ref.f32 / <dump>.cand.f32)
    #[arg(long)]
    dump_op: Option<String>,
}

fn fill_input(cli: &Cli, host: &mut [f32]) -> anyhow::Result<()> {
    if let Some(path) = &cli.input_file {
        let bytes = std::fs::read(path)?;
        anyhow::ensure!(
            bytes.len() == host.len() * 4,
            "{} holds {} f32s but the input needs {}",
            path.display(),
            bytes.len() / 4,
            host.len()
        );
        for (v, chunk) in host.iter_mut().zip(bytes.chunks_exact(4)) {
            *v = f32::from_le_bytes(chunk.try_into().unwrap());
        }
    } else {
        host.iter_mut()
            .enumerate()
            .for_each(|(i, v)| *v = (i as f32 * 0.9301).sin() * 0.5);
    }
    Ok(())
}

/// name, op type, logical shape, first-output values (empty when not captured)
type OpRow = (String, String, Vec<i32>, Vec<f32>);

fn run_model_per_op(cli: &Cli, forward: ForwardType) -> anyhow::Result<Vec<OpRow>> {
    let interpreter = Interpreter::from_file(&cli.model)?;
    let mut config = ScheduleConfig::new();
    config.set_type(forward);
    let mut backend_config = BackendConfig::new();
    backend_config.set_precision_mode(if forward == cli.reference {
        PrecisionMode::High
    } else {
        cli.precision
    });
    config.set_backend_config(backend_config);
    let mut session = interpreter.create_session(config)?;

    let input_names: Vec<String> = interpreter
        .inputs(&mut session)
        .iter_mut()
        .map(|x| x.name().to_string())
        .collect();

    if let Some(shape) = &cli.shape {
        for name in &input_names {
            interpreter.resize_tensor_by_name(&mut session, name, shape.as_slice())?;
        }
        interpreter.resize_session(&mut session);
    }

    for name in &input_names {
        let input = interpreter.input::<f32>(&mut session, name)?;
        let mut host = input.create_host_tensor_from_device(false);
        fill_input(cli, host.host_mut())?;
        input.copy_from_host_tensor(&host)?;
    }

    let rows: std::sync::Arc<std::sync::Mutex<Vec<OpRow>>> = Default::default();
    let sink_before = rows.clone();
    let watch = cli.dump_op.clone();
    let sink = rows.clone();
    interpreter.run_session_with_callback(
        &session,
        move |tensors, op| {
            // Capture the watched op's *input* too: if an op's inputs match
            // across backends but its output doesn't, the op itself is guilty.
            let name = op.name().to_string_lossy().into_owned();
            if watch.as_deref().is_some_and(|w| name.contains(w))
                && let Some(t) = tensors.first()
            {
                let shape = t.shape().as_ref().to_vec();
                if t.is_type_of::<f32>() && !shape.is_empty() && !shape.contains(&-1) {
                    let mut host =
                        Tensor::<Owned<f32>, Host>::new(shape.as_slice(), DimensionType::Caffe);
                    if t.copy_to_host_tensor(unsafe { host.as_any_tensor_mut() })
                        .is_ok()
                    {
                        sink_before.lock().unwrap().push((
                            format!("{name}#input"),
                            "input".into(),
                            shape,
                            host.host().to_vec(),
                        ));
                    }
                }
            }
            true
        },
        move |tensors, op| {
            let name = op.name().to_string_lossy().into_owned();
            let ty = op.type_name().to_string_lossy().into_owned();
            let mut shape = Vec::new();
            let mut data = Vec::new();
            if let Some(t) = tensors.first() {
                shape = t.shape().as_ref().to_vec();
                if t.is_type_of::<f32>() && !shape.is_empty() && !shape.contains(&-1) {
                    // Copy into an NCHW host tensor: both backends' packed
                    // layouts (e.g. NC4HW4) get unpacked to the same order,
                    // so values compare element-for-element.
                    let mut host =
                        Tensor::<Owned<f32>, Host>::new(shape.as_slice(), DimensionType::Caffe);
                    if t.copy_to_host_tensor(unsafe { host.as_any_tensor_mut() })
                        .is_ok()
                    {
                        data = host.host().to_vec();
                    }
                }
            }
            sink.lock().unwrap().push((name, ty, shape, data));
            true
        },
        true,
    )?;

    // The FFI side never drops the callback, so `rows` stays multiply-owned;
    // take the data out instead of unwrapping the Arc.
    let out = std::mem::take(&mut *rows.lock().unwrap());
    Ok(out)
}

fn compare_per_op(cli: &Cli, reference: &[OpRow], candidate: &[OpRow]) -> usize {
    use std::collections::HashMap;
    let mut cand_by_name: HashMap<&str, std::collections::VecDeque<usize>> = HashMap::new();
    for (i, (name, ..)) in candidate.iter().enumerate() {
        cand_by_name.entry(name).or_default().push_back(i);
    }

    let (mut matched, mut divergent, mut skipped) = (0usize, 0usize, 0usize);
    let mut worst: Vec<(f32, String, usize, usize)> = Vec::new();
    let mut stats = String::new();
    for (name, ty, shape, data) in reference {
        let Some(ci) = cand_by_name
            .get_mut(name.as_str())
            .and_then(|q| q.pop_front())
        else {
            skipped += 1;
            continue;
        };
        let (_, cty, _, cdata) = &candidate[ci];
        if data.is_empty() || cdata.is_empty() || data.len() != cdata.len() {
            skipped += 1;
            use std::fmt::Write;
            let _ = writeln!(stats, "skip\t-\t{ty}\t{name}\t{shape:?}");
            continue;
        }
        matched += 1;
        let mut max_abs = 0f32;
        let mut mismatched = 0usize;
        for (e, a) in data.iter().zip(cdata.iter()) {
            let d = (e - a).abs();
            max_abs = max_abs.max(d);
            if d > cli.tolerance {
                mismatched += 1;
            }
        }
        if mismatched > 0 {
            divergent += 1;
            if divergent <= 15 {
                println!(
                    "DIVERGE [{ty}/{cty}] {name} shape={shape:?}: max abs {max_abs:.6}, {mismatched}/{} over tol",
                    data.len()
                );
            }
            worst.push((
                max_abs,
                format!("[{ty}] {name} shape={shape:?}"),
                mismatched,
                data.len(),
            ));
        }
        {
            use std::fmt::Write;
            let _ = writeln!(stats, "{max_abs:.6}\t{mismatched}\t{ty}\t{name}\t{shape:?}");
        }
    }
    if let Some(path) = &cli.dump_stats {
        std::fs::write(path, &stats).expect("failed to write stats dump");
    }
    if let Some(pat) = &cli.dump_op {
        for (name, _, shape, data) in reference.iter().filter(|(n, ..)| n.contains(pat.as_str())) {
            if let Some((_, _, _, cdata)) = candidate.iter().find(|(n, ..)| n == name) {
                let to_bytes =
                    |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|f| f.to_le_bytes()).collect() };
                let stem = name.replace('/', "_");
                std::fs::write(format!("{stem}.ref.f32"), to_bytes(data)).unwrap();
                std::fs::write(format!("{stem}.cand.f32"), to_bytes(cdata)).unwrap();
                println!("dumped {name} shape={shape:?} to {stem}.{{ref,cand}}.f32");
            }
        }
    }
    worst.sort_by(|a, b| b.0.total_cmp(&a.0));
    if !worst.is_empty() {
        println!("\nworst divergences:");
        for (max_abs, what, mismatched, n) in worst.iter().take(15) {
            println!("  max abs {max_abs:.6} ({mismatched}/{n} over tol) {what}");
        }
    }
    println!(
        "\nper-op summary: {matched} matched, {divergent} divergent, {skipped} skipped/unmatched \
         (reference {} ops, candidate {} ops)",
        reference.len(),
        candidate.len()
    );
    divergent
}

fn run_model(cli: &Cli, forward: ForwardType) -> anyhow::Result<Vec<(String, Vec<f32>)>> {
    let interpreter = Interpreter::from_file(&cli.model)?;
    let mut config = ScheduleConfig::new();
    config.set_type(forward);
    let mut backend_config = BackendConfig::new();
    backend_config.set_precision_mode(if forward == cli.reference {
        PrecisionMode::High
    } else {
        cli.precision
    });
    config.set_backend_config(backend_config);
    let mut session = interpreter.create_session(config)?;

    let input_names: Vec<String> = interpreter
        .inputs(&mut session)
        .iter_mut()
        .map(|x| x.name().to_string())
        .collect();

    if let Some(shape) = &cli.shape {
        for name in &input_names {
            interpreter.resize_tensor_by_name(&mut session, name, shape.as_slice())?;
        }
        interpreter.resize_session(&mut session);
    }

    for name in &input_names {
        let input = interpreter.input::<f32>(&mut session, name)?;
        let mut host = input.create_host_tensor_from_device(false);
        // Deterministic input: either the provided file or a varied,
        // roughly zero-mean pattern so conv/matmul errors can't cancel
        // out the way an all-ones fill lets them.
        fill_input(cli, host.host_mut())?;
        input.copy_from_host_tensor(&host)?;
    }

    interpreter.run_session(&session)?;

    let mut outputs = Vec::new();
    for x in interpreter.outputs(&session).iter() {
        let name = x.name().to_string();
        let tensor = x.tensor::<f32>().map_err(|e| anyhow::anyhow!("{e:?}"))?;
        let host = tensor.create_host_tensor_from_device(true);
        outputs.push((name, host.host().to_vec()));
    }
    Ok(outputs)
}

fn main() -> anyhow::Result<()> {
    use clap::Parser;
    let cli = Cli::parse();

    if cli.per_op {
        let reference = run_model_per_op(&cli, cli.reference)?;
        let candidate = run_model_per_op(&cli, cli.candidate)?;
        let divergent = compare_per_op(&cli, &reference, &candidate);
        anyhow::ensure!(divergent == 0, "backend outputs diverge in {divergent} ops");
        println!("PARITY OK");
        return Ok(());
    }

    let reference = run_model(&cli, cli.reference)?;
    let candidate = run_model(&cli, cli.candidate)?;

    let mut failed = false;
    for ((name, expected), (_, actual)) in reference.iter().zip(candidate.iter()) {
        anyhow::ensure!(
            expected.len() == actual.len(),
            "output {name}: length mismatch ({} vs {})",
            expected.len(),
            actual.len()
        );
        let n = expected.len();
        let mut max_abs = 0f32;
        let mut sum_abs = 0f64;
        let mut mismatched = 0usize;
        for (e, a) in expected.iter().zip(actual.iter()) {
            let d = (e - a).abs();
            max_abs = max_abs.max(d);
            sum_abs += d as f64;
            if d > cli.tolerance {
                mismatched += 1;
            }
        }
        let argmax = |v: &[f32]| {
            v.iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, v)| (i, *v))
                .unwrap_or((0, f32::NAN))
        };
        let (ei, ev) = argmax(expected);
        let (ai, av) = argmax(actual);
        println!(
            "{name}: {n} elems, max abs diff {max_abs:.6}, mean abs diff {:.6}, {mismatched} ({:.2}%) over tol {}",
            sum_abs / n as f64,
            mismatched as f64 * 100.0 / n as f64,
            cli.tolerance,
        );
        println!(
            "  {:?}[..4] = {:?}\n  {:?}[..4] = {:?}\n  argmax: {:?} {ei} ({ev:.4}) vs {:?} {ai} ({av:.4})",
            cli.reference,
            &expected[..4.min(n)],
            cli.candidate,
            &actual[..4.min(n)],
            cli.reference,
            cli.candidate,
        );
        if mismatched > 0 {
            failed = true;
        }
    }
    anyhow::ensure!(!failed, "backend outputs diverge");
    println!("PARITY OK");
    Ok(())
}
