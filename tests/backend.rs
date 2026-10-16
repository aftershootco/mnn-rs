#![allow(unused_imports)]
pub mod common;
use common::*;
use mnn::ForwardType;
use tracing_test::traced_test;

#[cfg(feature = "coreml")]
#[test]
#[traced_test]
fn compare_cpu_and_coreml_outputs() {
    let net = mnn::Interpreter::from_file("tests/assets/realesr.mnn").unwrap();
    let cpu_config = ScheduleConfig::new();
    let mut coreml_config = ScheduleConfig::new();
    coreml_config.set_type(ForwardType::CoreML);
    let mut cpu_session = net.create_session(cpu_config).unwrap();
    let mut coreml_session = net.create_session(coreml_config).unwrap();
    net.inputs(&mut cpu_session).iter_mut().for_each(|mut x| {
        let tensor = x.tensor_mut::<f32>().expect("No tensor");
        tensor.fill(1.0f32).expect("Failed to fill tensor");
    });
    net.inputs(&mut coreml_session)
        .iter_mut()
        .for_each(|mut x| {
            let tensor = x.tensor_mut::<f32>().expect("No tensor");
            tensor.fill(1.0f32).expect("Failed to fill tensor");
        });

    net.run_session(&cpu_session).unwrap();
    net.run_session(&coreml_session).unwrap();

    let cpu_outputs = net.outputs(&cpu_session);
    let coreml_outputs = net.outputs(&coreml_session);

    cpu_outputs
        .iter()
        .zip(coreml_outputs.iter())
        .for_each(|(cpu, coreml)| {
            let cpu_tensor = cpu.tensor::<f32>().expect("No tensor");
            let coreml_tensor = coreml.tensor::<f32>().expect("No tensor");
            let cpu = cpu_tensor.create_host_tensor_from_device(true);
            let coreml = coreml_tensor.create_host_tensor_from_device(true);
            assert_eq!(cpu.host(), coreml.host());
        });
}
