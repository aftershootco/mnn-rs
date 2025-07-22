// use mnn::utils::*;
use mnn::*;
use std::path::PathBuf;

#[derive(Debug, clap::Parser, Clone)]
pub struct Cli {
    // image: PathBuf,
    model: PathBuf,
    // #[clap(short, long, default_value = "metal")]
    // forward: ForwardType,
    // #[clap(short, long, default_value = "high")]
    // precision: Modes,
    // #[clap(short = 'P', long, default_value = "high")]
    // power: Modes,
}

pub fn main() -> anyhow::Result<()> {
    use clap::Parser;
    let cli = Cli::parse();
    let mut interpreter = Interpreter::from_file(cli.model)?;

    let mut config = ScheduleConfig::new();
    config.set_type(ForwardType::CPU);
    let mut backend_config = BackendConfig::new();
    backend_config.set_precision_mode(PrecisionMode::High);
    backend_config.set_power_mode(PowerMode::High);
    config.set_backend_config(backend_config);

    let now = std::time::Instant::now();
    let session = interpreter.create_session(config)?;
    println!("create session time: {:?}", now.elapsed());
    let mut image = interpreter.input(&session, "image")?;
    let mut mask = interpreter.input(&session, "mask")?;
    let mut image_tensor = image.create_host_tensor_from_device(false);
    image_tensor.host_mut().fill(1.0f32);
    image.copy_from_host_tensor(image_tensor.view())?;
    let mut mask_tensor = mask.create_host_tensor_from_device(false);
    mask_tensor.host_mut().fill(0.7f32);
    let now = std::time::Instant::now();
    mask.copy_from_host_tensor(mask_tensor.view())?;
    println!("copy time: {:?}", now.elapsed());

    let output = interpreter.output(&session, "output")?;
    // image.copy_from_host_tensor(&unit_tensor)?;

    let now = std::time::Instant::now();
    interpreter.run_session(&session)?;
    output.wait(ffi::MapType::MAP_TENSOR_READ, true);
    println!("run time: {:?}", now.elapsed());

    let now = std::time::Instant::now();
    let output_tensor = output.create_host_tensor_from_device(true);
    println!("copy time: {:?}", now.elapsed());

    let out_vec = output_tensor.host().to_vec();
    let mut out_ppm = b"P6\n512 512\n255\n".to_vec();
    out_ppm.extend(out_vec.iter().map(|x: &f32| *x as u8));
    std::fs::write("output.ppm", out_ppm)?;

    Ok(())
}
