mod common;
use common::*;
use mnn::ForwardType;
use mnn::ScheduleConfig;

#[cfg(test)]
pub fn test_basic(backend: ForwardType) -> Result<()> {
    use mnn::BackendConfig;

    let mut net = mnn::Interpreter::from_bytes(Model::new())?;
    let mut config = ScheduleConfig::new();
    config.set_type(backend);
    config.set_backup_type(backend);
    let mut bc = BackendConfig::new();
    bc.set_memory_mode(mnn::MemoryMode::High);
    bc.set_precision_mode(mnn::PrecisionMode::High);
    bc.set_power_mode(mnn::PowerMode::High);
    config.set_backend_config(bc);
    let session = net.create_session(config)?;
    let inputs = net.inputs(&session);
    for input in inputs.iter() {
        println!("input: {:?}", input);
        input.tensor::<f32>()?.fill(0.0);
    }
    net.run_session(&session)?;
    let outputs = net.outputs(&session);
    for output in outputs.iter() {
        println!("output: {:?}", output);
        let tensor = output.tensor::<f32>()?;
        let shape = tensor.shape();
        assert_eq!(shape.as_ref(), [1, 3, 2048, 2048]);
    }
    Ok(())
}

#[test]
#[ignore = "takes too long"]
fn test_basic_cpu() {
    test_basic(ForwardType::CPU).unwrap();
}
#[cfg(feature = "metal")]
#[test]
fn test_basic_metal() {
    test_basic(ForwardType::Metal).unwrap();
}
#[cfg(feature = "opencl")]
#[test]
fn test_basic_opencl() {
    test_basic(ForwardType::OpenCL).unwrap();
}
#[cfg(feature = "coreml")]
#[test]
fn test_basic_coreml() {
    test_basic(ForwardType::CoreML).unwrap();
}
#[cfg(feature = "opengl")]
#[test]
fn test_basic_opengl() {
    test_basic(ForwardType::OpenGL).unwrap();
}
