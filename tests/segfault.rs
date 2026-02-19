/// This segfault on OpenCL backend if we print the tensorinfo
#[cfg(feature = "opencl")]
#[test]
fn test_segfault_case_1_() -> Result<(), Box<dyn std::error::Error>> {
    use mnn::*;
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

// #[test]
// #[ignore]
// pub fn test_resizing() {
//     use mnn::*;
//     let model = std::fs::read("tests/assets/resizing.mnn").expect("No resizing model");
//     let mut net = Interpreter::from_bytes(&model).unwrap();
//     let config = ScheduleConfig::default();
//     let mut session = net.create_session(config).unwrap();
//
//     loop {
//         let inputs = net.inputs(&session);
//         for tensor_info in inputs.iter() {
//             let mut tensor = unsafe { tensor_info.tensor_unresized::<f32>() }.unwrap();
//             let mut shape = tensor.shape().as_ref().to_vec();
//             dbg!(&shape);
//             shape.iter_mut().for_each(|v| {
//                 if *v == -1 {
//                     *v = 3;
//                 }
//             });
//             dbg!(&shape);
//             net.resize_tensor(tensor, &shape);
//         }
//         drop(inputs);
//
//         net.resize_session(&mut session);
//         let inputs = net.inputs(&session);
//         for tensor_info in inputs.iter() {
//             let tensor = tensor_info.tensor::<f32>().unwrap();
//             println!(
//                 "{:13}: {:>13}",
//                 tensor_info.name(),
//                 format!("{:?}", tensor.shape())
//             );
//             let mut host = tensor.create_host_tensor_from_device(false);
//             host.host_mut().fill(1.0);
//         }
//         drop(inputs);
//         net.run_session(&session).unwrap();
//     }
// }
