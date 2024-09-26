use mnn::*;

/// This segfault on OpenCL backend if we print the tensorinfo
#[cfg(feature = "opencl")]
#[test]
fn test_segfault_case_1_() -> Result<(), Box<dyn std::error::Error>> {
    let backend = ForwardType::OpenCL;
    let realesr = std::path::Path::new("tests/assets/realesr.mnn");
    use mnn::BackendConfig;

    let mut net = mnn::Interpreter::from_file(realesr)?;
    net.set_cache_file(realesr.with_extension("cache"), 128)?;
    let mut config = ScheduleConfig::new();
    config.set_type(backend);
    let mut session = net.create_session(config)?;
    net.update_cache_file(&mut session);

    net.inputs(&session).iter().for_each(|x| {
        let mut tensor = x.tensor::<f32>().expect("No tensor");
        // println!("{}: {:?}", x.name(), tensor.shape());
        println!("{:?}", x);
        tensor.fill(1.0f32);
    });
    net.run_session(&session)?;
    let outputs = net.outputs(&session);
    drop(outputs);
    drop(session);
    drop(net);
    Ok(())
}
