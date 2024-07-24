use std::path::PathBuf;

use mnn::{Device, DimensionType, Host, Interpreter, Tensor};
use mnn_sys::MNN::ScheduleConfig;
#[derive(Debug, clap::Parser, Clone)]
pub struct Cli {
    // image: PathBuf,
    model: PathBuf,
}
pub fn main() -> anyhow::Result<()> {
    use clap::Parser;
    let cli = Cli::parse();
    // let path = cli.image.to_str().unwrap();
    // let image =
    //     opencv::imgcodecs::imread(path, opencv::imgcodecs::ImreadModes::IMREAD_COLOR.into())?;
    let mut interpreter = Interpreter::from_file(cli.model)?;
    interpreter.set_session_mode(mnn::ffi::MNN::Interpreter_SessionMode::Session_Release);
    interpreter.create_session()?;

    let gpu_tensor: Tensor<Device, f32> =
        Tensor::create_device(&[16, 256, 256, 3], DimensionType::NHWC);
    gpu_tensor.print_shape();
    Ok(())
}
