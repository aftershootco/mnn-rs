use ffi::MNNForwardType;
use mnn::*;
use std::path::PathBuf;

#[derive(Debug, clap::Parser, Clone)]
pub struct Cli {
    // image: PathBuf,
    model: PathBuf,
}
pub fn main() -> anyhow::Result<()> {
    use clap::Parser;
    let cli = Cli::parse();
    let mut interpreter = Interpreter::from_file(cli.model)?;

    let mut config = ScheduleConfig::new();
    config.set_type(MNNForwardType::MNN_FORWARD_VULKAN);
    let mut backend_config = BackendConfig::new();
    backend_config.set_precision_mode(PrecisionMode::Precision_High);
    backend_config.set_power_mode(PowerMode::Power_High);
    config.set_backend_config(&backend_config);

    let session = interpreter.create_session(&mut config)?;
    let mut image = interpreter.get_input(&session, "image")?;
    let mut mask = interpreter.get_input(&session, "mask")?;
    let unit_tensor_data = vec![1.0f32; 1 * 3 * 512 * 512];
    let mut image_tensor = image.create_host_tensor_from_device(false);
    image_tensor.host_mut().copy_from_slice(&unit_tensor_data);
    image.copy_from_host_tensor(&image_tensor)?;
    let mut mask_tensor = mask.create_host_tensor_from_device(false);
    mask_tensor.host_mut().fill(0.7f32);
    let now = std::time::Instant::now();
    mask.copy_from_host_tensor(&mask_tensor)?;
    println!("copy time: {:?}", now.elapsed());

    // image.copy_from_host_tensor(&unit_tensor)?;

    interpreter.run_session(&session)?;
    let output = interpreter.get_output(&session, "output")?;
    let output_tensor = output.create_host_tensor_from_device(true);
    drop(output);
    let output = interpreter.get_output(&session, "output")?;

    let out_vec = output_tensor.host::<f32>().to_vec();
    let mut out_ppm = b"P6\n512 512\n255\n".to_vec();
    out_ppm.extend(out_vec.iter().map(|x| *x as u8));
    std::fs::write("output.ppm", out_ppm)?;

    Ok(())
}
