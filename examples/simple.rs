use mnn::ffi::MNNForwardType;
use mnn::*;
use std::path::PathBuf;

#[derive(Debug, clap::Parser, Clone)]
pub struct Cli {
    // image: PathBuf,
    model: PathBuf,
    #[clap(short, long, default_value = "metal")]
    forward: ForwardType,
    #[clap(short, long, default_value = "high")]
    precision: Modes,
    #[clap(short, long, default_value = "high")]
    power: Modes,
}

#[derive(clap::ValueEnum, Debug, Clone)]
pub enum ForwardType {
    CPU,
    Metal,
    Vulkan,
}

impl ForwardType {
    fn to_forward_type(&self) -> MNNForwardType {
        match self {
            ForwardType::CPU => MNNForwardType::MNN_FORWARD_CPU,
            ForwardType::Metal => MNNForwardType::MNN_FORWARD_METAL,
            ForwardType::Vulkan => MNNForwardType::MNN_FORWARD_VULKAN,
        }
    }
}

#[derive(clap::ValueEnum, Debug, Clone)]
pub enum Modes {
    Low,
    Normal,
    High,
}

impl Modes {
    fn to_precision_mode(&self) -> PrecisionMode {
        match self {
            Modes::Low => PrecisionMode::Precision_Low,
            Modes::Normal => PrecisionMode::Precision_Normal,
            Modes::High => PrecisionMode::Precision_High,
        }
    }
    fn to_power_mode(&self) -> PowerMode {
        match self {
            Modes::Low => PowerMode::Power_Low,
            Modes::Normal => PowerMode::Power_Normal,
            Modes::High => PowerMode::Power_High,
        }
    }
}
// fn num_to_type(num: u32) -> MNNForwardType {
// }

pub fn main() -> anyhow::Result<()> {
    use clap::Parser;
    let cli = Cli::parse();
    let mut interpreter = Interpreter::from_file(cli.model)?;

    let mut config = ScheduleConfig::new();
    config.set_type(cli.forward.to_forward_type());
    let mut backend_config = BackendConfig::new();
    backend_config.set_precision_mode(cli.precision.to_precision_mode());
    backend_config.set_power_mode(cli.power.to_power_mode());
    config.set_backend_config(&backend_config);

    let now = std::time::Instant::now();
    let session = interpreter.create_session(&mut config)?;
    println!("create session time: {:?}", now.elapsed());
    let mut image = interpreter.get_input(&session, "image")?;
    let mut mask = interpreter.get_input(&session, "mask")?;
    let mut image_tensor = image.create_host_tensor_from_device(false);
    image_tensor.host_mut().fill(1.0f32);
    image.copy_from_host_tensor(&image_tensor)?;
    let mut mask_tensor = mask.create_host_tensor_from_device(false);
    mask_tensor.host_mut().fill(0.7f32);
    let now = std::time::Instant::now();
    mask.copy_from_host_tensor(&mask_tensor)?;
    println!("copy time: {:?}", now.elapsed());

    let output = interpreter.get_output(&session, "output")?;
    // image.copy_from_host_tensor(&unit_tensor)?;

    let now = std::time::Instant::now();
    interpreter.run_session(&session)?;
    output.wait(ffi::MapType::MAP_TENSOR_READ, true);
    println!("run time: {:?}", now.elapsed());

    let now = std::time::Instant::now();
    let output_tensor = output.create_host_tensor_from_device(true);
    println!("copy time: {:?}", now.elapsed());

    let out_vec = output_tensor.host::<f32>().to_vec();
    let mut out_ppm = b"P6\n512 512\n255\n".to_vec();
    out_ppm.extend(out_vec.iter().map(|x| *x as u8));
    std::fs::write("output.ppm", out_ppm)?;

    Ok(())
}
