pub mod common;
use common::*;
use mnn::ForwardType;

#[test]
#[ignore = "Doesn't work on ci"]
fn test_basic_cpu() {
    test_basic(ForwardType::CPU).unwrap();
}
#[cfg(feature = "metal")]
#[test]
#[ignore = "Doesn't work on ci"]
fn test_basic_metal() {
    test_basic(ForwardType::Metal).unwrap();
}
#[cfg(feature = "opencl")]
#[test]
#[ignore = "Doesn't work on ci"]
fn test_basic_opencl() -> Result<(), Box<dyn std::error::Error>> {
    let backend = ForwardType::OpenCL;
    let realesr = std::path::Path::new("tests/assets/realesr.mnn");

    let mut net = mnn::Interpreter::from_file(realesr)?;
    net.set_cache_file(realesr.with_extension("cache"), 128)?;
    let mut config = ScheduleConfig::new();
    config.set_type(backend);
    let mut session = net.create_session(config)?;
    net.update_cache_file(&mut session)?;

    net.inputs(&session).iter().for_each(|x| {
        let mut tensor = x.tensor::<f32>().expect("No tensor");
        println!("{}: {:?}", x.name(), tensor.shape());
        tensor.fill(1.0f32);
    });
    net.run_session(&session)?;
    let outputs = net.outputs(&session);
    outputs.iter().for_each(|x| {
        let tensor = x.tensor::<f32>().expect("No tensor");
        tensor.wait(ffi::MapType::MAP_TENSOR_READ, true);
        println!("Waiting for tensor: {}", x.name());
        println!("{}: {:?}", x.name(), tensor.shape());
        // let _ = tensor.create_host_tensor_from_device(true);
    });

    // drop(outputs);
    // drop(session);
    // drop(net);
    Ok(())
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

#[test]
#[ignore = "takes too long and unreliable on CI"]
fn test_multi_path_cpu_cpu() {
    test_multipath_session(ForwardType::CPU, ForwardType::CPU).unwrap();
}

// #[cfg(feature = "opencl")]
// #[test]
// fn test_multi_path_opencl_cpu() {
//     test_multipath_session(ForwardType::OpenCL, ForwardType::CPU).unwrap();
// }
