use console::Term;
use error_stack::*;
use indicatif::{MultiProgress, ProgressBar};
use mnn::ScheduleConfig;
use std::{
    collections::BTreeMap,
    io::IsTerminal,
    path::{Path, PathBuf},
    time::Duration,
};
use thiserror::Error;
use tracing_subscriber::{layer::SubscriberExt as _, util::SubscriberInitExt as _};
#[derive(Debug, Clone, Error, Copy)]
#[error("BenchError: Failed to bench")]
pub struct BenchError;
use clap::*;

pub trait ResultExtCC: ResultExt + Sized {
    #[track_caller]
    fn cc<C: Context>(self, context: C) -> core::result::Result<Self::Ok, Report<C>> {
        self.change_context(context)
    }
}

impl<T> ResultExtCC for T where T: ResultExt {}

#[derive(Debug, Clone, Parser)]
pub struct Generate {
    #[arg(required = true)]
    models: Vec<PathBuf>,
    // Always generate with cpu by default
    #[arg(short, long, default_value = "cpu")]
    forward: mnn::ForwardType,
    #[arg(short, long, default_value = "high")]
    power: mnn::PowerMode,
    #[arg(short, long, default_value = "high")]
    precision: mnn::PrecisionMode,
    #[arg(short, long, default_value = "high")]
    memory: mnn::MemoryMode,
    // #[arg(flatten)]
    // output_types: Vec<TypedOutput>,
    #[arg(short, long)]
    output_type: DataType,
}

#[derive(Debug, Clone, Args)]
pub struct TypedOutput {
    name: String,
    data_type: DataType,
}

#[derive(Debug, Clone, Parser)]
pub struct Cli {
    #[clap(subcommand)]
    subcommand: Subcommand,
    #[command(flatten)]
    verbose: clap_verbosity_flag::Verbosity,
}

#[derive(Debug, Clone, Subcommand)]
pub enum Subcommand {
    Bench(Bench),
    Generate(Generate),
    Completions(Completions),
}
#[derive(Debug, Clone, Parser)]
pub struct Completions {
    #[arg(short, long)]
    pub shell: clap_complete::Shell,
}

#[derive(Debug, Clone, Parser)]
pub struct Bench {
    #[arg(required = true)]
    models: Vec<PathBuf>,
    #[command(flatten)]
    sc_items: ScheduleConfigItems,
    #[arg(short, long, default_value = "10")]
    warmup: u8,
    #[arg(short, long)]
    output: Option<PathBuf>,
    /// Run in exec mode i.e. run the self binary with the given arguments individually. This
    /// provides a way to bypass segmentation faults in the library.
    #[arg(short, long)]
    exec: bool,
}

#[derive(Default, Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Config {
    inputs: BTreeMap<String, PathBuf>,
    outputs: BTreeMap<String, ConfigData>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ConfigData {
    data_type: DataType,
    path: PathBuf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize, ValueEnum)]
pub enum DataType {
    // Float16,
    Float32,
    Int32,
    Int64,
    Int8,
    Uint8,
}

impl DataType {
    pub fn mas(&self, lhs: &[u8], rhs: &[u8]) -> f64 {
        match self {
            Self::Float32 => Self::mean_absolute_error_bytes::<f32>(lhs, rhs),
            Self::Int32 => Self::mean_absolute_error_bytes::<i32>(lhs, rhs),
            Self::Int64 => Self::mean_absolute_error_bytes::<i64>(lhs, rhs),
            Self::Int8 => Self::mean_absolute_error_bytes::<i8>(lhs, rhs),
            Self::Uint8 => Self::mean_absolute_error_bytes::<u8>(lhs, rhs),
        }
    }

    pub fn mean_absolute_error_bytes<
        T: core::ops::Sub<Output = T>
            + PartialOrd
            + Copy
            + core::ops::Add<Output = T>
            + num::cast::AsPrimitive<f64>
            + bytemuck::Pod,
    >(
        lhs: &[u8],
        rhs: &[u8],
    ) -> f64 {
        let lhs = bytemuck::cast_slice(lhs);
        let rhs = bytemuck::cast_slice(rhs);
        Self::mean_absolute_error::<T>(lhs, rhs)
    }

    pub fn mean_absolute_error<
        T: core::ops::Sub<Output = T>
            + PartialOrd
            + Copy
            + core::ops::Add<Output = T>
            + num::cast::AsPrimitive<f64>,
    >(
        lhs: impl AsRef<[T]>,
        rhs: impl AsRef<[T]>,
    ) -> f64 {
        let (sum, count) = lhs
            .as_ref()
            .iter()
            .zip(rhs.as_ref())
            .map(|(&l, &r)| if l > r { l - r } else { r - l })
            .fold((0f64, 0usize), |(acc, count), x| (acc + x.as_(), count + 1));
        sum / count as f64
    }
}

impl Config {
    pub fn find(model: impl AsRef<Path>) -> Result<Self> {
        let model = model.as_ref();
        let config = model.with_extension("json");
        let config = std::fs::read(config).cc(BenchError)?;
        let config: Config = serde_json::from_slice(&config).cc(BenchError)?;
        Ok(config)
    }
}

#[derive(Debug, Clone, Args)]
pub struct ScheduleConfigItems {
    /// Comma separated list of forward types (cpu / opencl / metal / coreml)
    #[arg(short, long, value_delimiter = ',', num_args= 1.., default_value = "cpu")]
    forward: Vec<mnn::ForwardType>,
    /// Comma separated list of power modes (low / high / normal)
    #[arg(short = 'P', long,value_delimiter = ',', num_args= 1.., default_value = "normal")]
    power: Vec<mnn::PowerMode>,
    /// Comma separated list of precision modes (low / high / normal)
    #[arg(short, long,value_delimiter = ',', num_args= 1.., default_value = "normal")]
    precision: Vec<mnn::PrecisionMode>,
    /// Comma separated list of memory modes (low / high / normal)
    #[arg(short, long,value_delimiter = ',', num_args= 1.., default_value = "normal")]
    memory: Vec<mnn::MemoryMode>,
}

pub struct ScheduleConfigItem {
    pub forward: mnn::ForwardType,
    pub power: mnn::PowerMode,
    pub precision: mnn::PrecisionMode,
    pub memory: mnn::MemoryMode,
}

impl ScheduleConfigItem {
    pub fn new(
        forward: mnn::ForwardType,
        power: mnn::PowerMode,
        precision: mnn::PrecisionMode,
        memory: mnn::MemoryMode,
    ) -> Self {
        Self {
            forward,
            power,
            precision,
            memory,
        }
    }

    pub fn into_schedule_config(self) -> ScheduleConfig {
        let mut sc = mnn::ScheduleConfig::new();
        let mut bc = mnn::BackendConfig::new();
        bc.set_power_mode(self.power);
        bc.set_precision_mode(self.precision);
        bc.set_memory_mode(self.memory);
        sc.set_type(self.forward).set_backend_config(bc);
        sc
    }
}

impl ScheduleConfigItems {
    pub fn is_empty(&self) -> bool {
        self.forward.is_empty()
            || self.power.is_empty()
            || self.precision.is_empty()
            || self.memory.is_empty()
    }

    pub fn is_single(&self) -> bool {
        self.combinations() == 1
    }

    pub fn combinations(&self) -> usize {
        self.forward.len() * self.power.len() * self.precision.len() * self.memory.len()
    }
}

impl IntoIterator for ScheduleConfigItems {
    type Item = ScheduleConfigItem;
    type IntoIter = std::vec::IntoIter<ScheduleConfigItem>;

    fn into_iter(self) -> Self::IntoIter {
        let outputs: Vec<ScheduleConfigItem> = self
            .forward
            .iter()
            .map(|f| {
                self.power.iter().map(|p| {
                    self.precision.iter().map(|pr| {
                        self.memory
                            .iter()
                            .map(|m| ScheduleConfigItem::new(*f, *p, *pr, *m))
                    })
                })
            })
            .flatten()
            .flatten()
            .flatten()
            .collect();
        outputs.into_iter()
    }
}
type Result<T, E = Report<BenchError>> = core::result::Result<T, E>;

#[derive(Debug, serde::Serialize)]
pub struct Metrics {
    pub model: PathBuf,
    pub metrics: Vec<Metric>,
}

#[derive(Debug)]
pub struct Metric {
    pub memory: f32,                 // in MiB
    pub flops: f32,                  // in Mflops
    pub initial_load_time: Duration, // in ms
    pub cached_load_time: Duration,  // in ms
    pub inference_time: Duration,    // in ms
    pub schedule_config: ScheduleConfig,
    pub outputs: BTreeMap<String, f64>, // mean absolute error
}

impl serde::Serialize for Metric {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct as _;
        let mut state = serializer.serialize_struct("Metric", 6)?;
        state.serialize_field("memory", &format!("{:.0}MiB", self.memory))?;
        state.serialize_field("flops", &format!("{:.0}M", self.flops))?;
        state.serialize_field(
            "initial_load_time",
            &format!("{}ms", self.initial_load_time.as_millis()),
        )?;
        state.serialize_field(
            "cached_load_time",
            &format!("{}ms", self.cached_load_time.as_millis()),
        )?;
        state.serialize_field(
            "inference_time",
            &format!("{}ms", self.inference_time.as_millis()),
        )?;
        state.serialize_field("schedule_config", &self.schedule_config)?;
        state.serialize_field("outputs", &self.outputs)?;
        state.end()
    }
}

pub fn main() -> Result<()> {
    let cli = Cli::parse();
    // let cli = Bench::parse();
    // let indicatif_layer = IndicatifLayer::new();
    tracing_subscriber::registry()
        .with(cli.verbose.tracing_level_filter())
        // .with(tracing_subscriber::fmt::layer().with_writer(Term::stderr))
        .init();

    match cli.subcommand {
        Subcommand::Bench(cli) => bench_main(cli)?,
        Subcommand::Generate(cli) => generate_main(cli)?,
        Subcommand::Completions(cli) => {
            use clap_complete::aot::{generate, Generator, Shell};

            let shell = cli.shell;
            let mut cmd = Cli::command();
            let name = cmd.get_name().to_string();
            generate(shell, &mut cmd, name, &mut std::io::stdout());
        }
    }

    Ok(())
}

pub fn generate_main(cli: Generate) -> Result<()> {
    for model in cli.models {
        let mut cfg = Config {
            inputs: Default::default(),
            outputs: Default::default(),
        };
        let mut net = mnn::Interpreter::from_file(&model).cc(BenchError)?;
        let sc = ScheduleConfig::new()
            .with_type(cli.forward)
            .with_backend_config(
                mnn::BackendConfig::new()
                    .with_power_mode(cli.power)
                    .with_precision_mode(cli.precision)
                    .with_memory_mode(cli.memory),
            );
        let session = net.create_session(sc).cc(BenchError)?;
        let inputs = net.inputs(&session);
        for input in &inputs {
            let model_name = model
                .file_stem()
                .expect("Failed to get model name")
                .to_string_lossy();
            let name = format!("{}_input_{}.bin", model_name, input.name());
            let path = model.with_file_name(name);
            let tensor = input.raw_tensor().create_host_tensor_from_device(false);
            unsafe {
                tensor.unchecked_host_bytes().fill(1);
                std::fs::write(&path, tensor.unchecked_host_bytes()).cc(BenchError)?;
            }
            cfg.inputs.insert(
                input.name().to_string(),
                dunce::canonicalize(path).cc(BenchError)?,
            );
        }
        drop(inputs);

        net.run_session(&session);

        let outputs = net.outputs(&session);
        for output in &outputs {
            let model_name = model
                .file_stem()
                .expect("Failed to get model name")
                .to_string_lossy();
            let name = format!("{}_output_{}.bin", model_name, output.name());
            let path = model.with_file_name(name);
            let tensor = output.raw_tensor();
            unsafe {
                std::fs::write(
                    &path,
                    tensor
                        .create_host_tensor_from_device(true)
                        .unchecked_host_bytes(),
                )
                .cc(BenchError)?;
            }
            cfg.outputs.insert(
                output.name().to_string(),
                ConfigData {
                    data_type: cli.output_type,
                    path: dunce::canonicalize(path).cc(BenchError)?,
                },
            );
        }
        std::fs::write(
            model.with_extension("json"),
            serde_json::to_string_pretty(&cfg).cc(BenchError)?,
        )
        .cc(BenchError)?;
    }
    Ok(())
}

pub fn bench_main(cli: Bench) -> Result<()> {
    let multi_progress = indicatif::MultiProgress::new();
    let output = if !cli.exec {
        let results = bench_all(cli.models.iter(), cli.sc_items, cli.warmup, &multi_progress);
        serde_json::to_string_pretty(&results).cc(BenchError)?
    } else {
        let results = exec_bench_all(cli.models.iter(), cli.sc_items, cli.warmup, &multi_progress)?;
        serde_json::to_string_pretty(&results).cc(BenchError)?
    };
    use std::io::Write;
    if let Some(out_f) = cli.output {
        std::fs::File::create(out_f)
            .cc(BenchError)?
            .write_all(output.as_bytes())
            .cc(BenchError)?;
    } else {
        Term::stdout().write_all(output.as_bytes()).cc(BenchError)?;
    }
    Ok(())
}

pub fn exec_bench_all<'a>(
    models: impl Iterator<Item = &'a PathBuf>,
    sc_items: ScheduleConfigItems,
    warmup: u8,
    mp: &MultiProgress,
) -> Result<Vec<Result<serde_json::Value>>> {
    let self_exe = std::env::current_exe().cc(BenchError)?;
    let result: Vec<Result<serde_json::Value>> = models
        .map(|m| {
            let pb = indicatif::ProgressBar::new(sc_items.combinations() as u64)
                .with_prefix(format!("{}", m.file_name().unwrap().to_string_lossy()))
                .with_style(
                    indicatif::ProgressStyle::default_bar()
                        .template("{prefix} {bar:80} {pos}/{len} {msg}")
                        .expect("Failed to build progress bar style"),
                );
            mp.insert(0, pb.clone());
            sc_items
                .clone()
                .into_iter()
                .map({
                    |sc| {
                        pb.set_message(format!(
                            "{:?}:power->{:?}:precision->{:?}:memory->{:?}",
                            sc.forward, sc.power, sc.precision, sc.memory
                        ));
                        let out = exec_bench(&self_exe, warmup, sc, m, &mp);
                        pb.inc(1);
                        out
                    }
                })
                .collect::<Vec<_>>()
        })
        .flatten()
        .collect();
    Ok(result)
}

pub fn exec_bench(
    exec: &Path,
    w: u8,
    sc: ScheduleConfigItem,
    model: impl AsRef<Path>,
    mp: &MultiProgress,
) -> Result<serde_json::Value> {
    let mut child = std::process::Command::new(exec)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .arg("bench")
        .arg(model.as_ref())
        .arg("--memory")
        .arg(sc.memory.to_str())
        .arg("--power")
        .arg(sc.power.to_str())
        .arg("--precision")
        .arg(sc.precision.to_str())
        .arg("--forward")
        .arg(sc.forward.to_str())
        .arg("--warmup")
        .arg(w.to_string())
        .spawn()
        .cc(BenchError)?;
    let child_stderr = child.stderr.take().expect("Failed to get stderr");
    let child_stdout = child.stdout.take().expect("Failed to get stdout");
    let progress = p_read(child_stderr);
    progress.enable_steady_tick(Duration::from_millis(200));
    mp.insert(0, progress.clone());
    let output = child.wait().cc(BenchError)?;
    if !output.success() {
        return Err(Report::new(BenchError)
            .attach_printable(format!("Failed to execute {exec}", exec = exec.display())));
    }
    progress.finish_and_clear();
    let metrics = serde_json::from_reader(child_stdout).cc(BenchError)?;
    Ok(metrics)
}

pub fn bench_all(
    models: impl Iterator<Item = impl AsRef<Path>>,
    sc_items: ScheduleConfigItems,
    warmup: u8,
    multi_progress: &MultiProgress,
) -> Vec<Result<Metrics>> {
    let result: Vec<Result<Metrics>> = models
        .map(|m| -> Result<Metrics> {
            // Check create_session_time without cache
            let pb = indicatif::ProgressBar::new(sc_items.combinations() as u64)
                .with_prefix(format!(
                    "{}",
                    m.as_ref().file_name().unwrap().to_string_lossy()
                ))
                .with_style(if sc_items.is_single() {
                    indicatif::ProgressStyle::default_bar()
                        .template("{prefix} {msg}")
                        .expect("Failed to build progress bar style")
                } else {
                    indicatif::ProgressStyle::default_bar()
                        .template("{prefix} {bar:80} {pos}/{len} {msg}")
                        .expect("Failed to build progress bar style")
                });

            multi_progress.add(pb.clone());
            let metrics = sc_items
                .clone()
                .into_iter()
                .map(|sc| {
                    pb.set_message(format!(
                        "{:?}:power->{:?}:precision->{:?}:memory->{:?}",
                        sc.forward, sc.power, sc.precision, sc.memory
                    ));
                    let o = bench(
                        warmup,
                        sc.into_schedule_config(),
                        m.as_ref(),
                        &multi_progress,
                    )
                    .cc(BenchError);
                    pb.inc(1);
                    o
                })
                .collect::<Result<Vec<Metric>>>()
                .cc(BenchError)?;
            Ok(Metrics {
                model: dunce::canonicalize(m).cc(BenchError)?,
                metrics,
            })
        })
        .collect();
    result
}

// #[tracing::instrument(skip(model))]
pub fn bench(
    w: u8,
    sc: ScheduleConfig,
    model: impl AsRef<Path>,
    mp: &MultiProgress,
) -> Result<Metric> {
    let bar = indicatif::ProgressBar::new_spinner();
    mp.insert(0, bar.clone());
    bar.enable_steady_tick(Duration::from_millis(300));
    let not_terminal = !std::io::stdout().is_terminal();

    tracing::info!("Benching {:?}", sc);
    let mut net = mnn::Interpreter::from_file(&model).cc(BenchError)?;

    bar.set_message("Creating session without cache");
    not_terminal.then(|| eprintln!("Creating session without cache"));
    let (mut uncached, initial_load_time) = timeit(|| {
        tracing::trace!("Creating session without cache");
        net.create_session(sc.clone())
    })
    .cc(BenchError)?;
    let temp_file = temp_file_path()?;
    net.set_cache_file(&temp_file, 128).cc(BenchError)?;
    net.update_cache_file(&mut uncached).cc(BenchError)?;
    drop(uncached);
    drop(net);
    let mut net = mnn::Interpreter::from_file(&model).cc(BenchError)?;
    net.set_cache_file(&temp_file, 128).cc(BenchError)?;
    bar.set_message("Creating session with cache");
    not_terminal.then(|| eprintln!("Creating session with cache"));
    let (session, cached_load_time) = timeit(|| {
        tracing::trace!("Creating session with cache {temp_file:?}");
        net.create_session(sc.clone())
    })
    .cc(BenchError)?;
    for c in 0..w {
        bar.set_message(format!("Warming up {c}"));
        not_terminal.then(|| eprintln!("Warming up {c}"));
        net.run_session(&session).cc(BenchError)?;
    }
    let config = Config::find(&model).cc(BenchError).unwrap_or_default();
    for (name, path) in config.inputs.iter() {
        let input = std::fs::read(path).cc(BenchError)?;
        bar.set_message(format!("Setting input {name}"));
        not_terminal.then(|| eprintln!("Setting input {name}"));
        unsafe {
            net.raw_input(&session, name)
                .cc(BenchError)?
                .create_host_tensor_from_device(false)
                .unchecked_host_bytes()
                .copy_from_slice(&input);
        }
    }
    let (_, inference_time) = timeit(|| -> Result<()> {
        bar.set_message("Running session");
        not_terminal.then(|| eprintln!("Running session"));
        net.run_session(&session).cc(BenchError)?;
        net.wait(&session);
        Ok(())
    })
    .cc(BenchError)?;

    let mut outputs = BTreeMap::new();
    for (name, path) in config.outputs.iter() {
        bar.set_message(format!("Checking output {name}"));
        not_terminal.then(|| eprintln!("Checking output {name}"));
        let output = unsafe {
            net.raw_output(&session, name)
                .cc(BenchError)?
                .create_host_tensor_from_device(true)
                .unchecked_host_bytes()
                .to_vec()
        };
        if let Some(cd) = config.outputs.get(name) {
            let expected = std::fs::read(&cd.path).cc(BenchError)?;
            let mas = cd.data_type.mas(&output, &expected);
            outputs.insert(name.clone(), mas);
        }
    }
    let memory = net.memory(&session).cc(BenchError)?;
    let flops = net.flops(&session).cc(BenchError)?;
    temp_file.close().cc(BenchError)?;
    Ok(Metric {
        schedule_config: sc,
        memory,
        flops,
        initial_load_time,
        cached_load_time,
        inference_time,
        outputs,
    })
}

pub fn timeit<F: FnOnce() -> Result<T, E>, T, E>(f: F) -> Result<(T, Duration), E> {
    let start = std::time::Instant::now();
    let result = f()?;
    let duration = start.elapsed();
    Ok((result, duration))
}

pub fn temp_file_path() -> Result<tempfile::TempPath> {
    Ok(tempfile::NamedTempFile::new()
        .cc(BenchError)?
        .into_temp_path())
}

pub fn p_read(reader: impl std::io::Read + Send + Sync + 'static) -> ProgressBar {
    let bar = ProgressBar::new_spinner().with_style(
        indicatif::ProgressStyle::default_bar()
            .template("{spinner} {msg}")
            .expect("Failed to build progress bar style"),
    );
    let bar_ = bar.clone();

    std::thread::spawn(move || {
        use std::io::BufRead;
        let mut reader = std::io::BufReader::new(reader);
        let mut buffer = String::new();
        while reader
            .read_line(&mut buffer)
            .cc(BenchError)
            .expect("Failed to read line")
            > 0
        {
            buffer.ends_with('\n').then(|| buffer.pop());
            bar.set_message(buffer.clone());
            buffer.clear();
            std::thread::sleep(Duration::from_millis(100));
            if bar.is_finished() {
                break;
            }
        }
    });
    bar_
}
