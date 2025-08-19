mod common;
use common::*;

#[test]
pub fn test_resizing() -> Result<()> {
    let model = std::fs::read("tests/assets/resizing.mnn").expect("No resizing model");
    let mut net = Interpreter::from_bytes(&model).unwrap();
    net.set_cache_file("resizing.cache", 128)?;
    let config = ScheduleConfig::default();
    // #[cfg(feature = "opencl")]
    // config.set_type(ForwardType::OpenCL);
    let mut session = net.create_session(config).unwrap();
    net.update_cache_file(&mut session)?;

    let now = std::time::Instant::now();
    let mask = unsafe { net.input_unresized::<f32>(&session, "mask") }?;
    dbg!(mask.shape());
    dbg!(mask.shape());
    dbg!(mask.shape());
    dbg!(mask.shape());
    net.resize_tensor(mask, [2048, 2048]);

    let og = unsafe { net.input_unresized::<f32>(&session, "original") }?;
    net.resize_tensor(og, [2048, 2048, 3]);

    let pain = unsafe { net.input_unresized::<f32>(&session, "inpainted") }?;
    net.resize_tensor(pain, [2048, 2048, 3]);
    // drop(pain);

    net.resize_session(&mut session);
    let inputs = net.inputs(&session);
    for tensor_info in inputs.iter() {
        let tensor = tensor_info.tensor::<f32>().unwrap();
        println!(
            "{:13}: {:>13}",
            tensor_info.name(),
            format!("{:?}", tensor.shape())
        );
        let mut host = tensor.create_host_tensor_from_device(false);
        host.host_mut().fill(1.0);
    }
    drop(inputs);
    net.run_session(&session).unwrap();
    println!("{:?}", now.elapsed());
    Ok(())
}
