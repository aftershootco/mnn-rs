use ffi::{BackendConfig, ScheduleConfig};
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
    config.type_ = ffi::MNNForwardType::MNN_FORWARD_METAL;
    let mut backend_config = BackendConfig::new();
    backend_config.precision = ffi::PrecisionMode::Precision_High;
    backend_config.power = ffi::PowerMode::Power_High;

    config.backendConfig = core::ptr::addr_of!(backend_config).cast_mut();
    let session = interpreter.create_session(&config)?;
    let inputs = interpreter.get_inputs(&session);
    let outputs = interpreter.get_outputs(&session);
    let mut image = inputs
        .iter()
        .find(|x| x.name() == "image")
        .expect("No input named image")
        .tensor();
    let mut mask = inputs
        .iter()
        .find(|x| x.name() == "mask")
        .expect("No input named mask")
        .tensor();
    let unit_tensor_data = vec![1.0f32; 1 * 3 * 512 * 512];
    let mut image_tensor = image.create_host_tensor_from_device(false);
    image_tensor.host_mut().copy_from_slice(&unit_tensor_data);
    image.copy_from_host_tensor(&image_tensor)?;
    let mut mask_tensor = mask.create_host_tensor_from_device(false);
    mask_tensor.host_mut().fill(0.7f32);
    mask.copy_from_host_tensor(&mask_tensor)?;
    // mask.host_mut::<f32>().fill(0.7f32);

    // image.copy_from_host_tensor(&unit_tensor)?;

    interpreter.run_session(&session)?;
    let output = outputs
        .iter()
        .find(|x| x.name() == "output")
        .expect("Not output named output")
        .tensor();
    let output_tensor = output.create_host_tensor_from_device(true);

    let out_vec = output_tensor.host::<f32>().to_vec();
    let mut out_ppm = b"P6\n512 512\n255\n".to_vec();
    out_ppm.extend(out_vec.iter().map(|x| *x as u8));
    std::fs::write("output.ppm", out_ppm)?;

    Ok(())
}
