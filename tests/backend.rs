pub mod common;
use common::*;
use mnn::ForwardType;
use tracing_test::traced_test;

#[test]
#[traced_test]
fn compare_cpu_and_coreml_outputs() {
    let mut net = mnn::Interpreter::from_file("tests/assets/realesr.mnn").unwrap();
    let cpu_config = ScheduleConfig::new();
    let mut coreml_config = ScheduleConfig::new();
    let mut bc = BackendConfig::new();
    coreml_config.set_type(ForwardType::CoreML);
    let cpu_session = net.create_session(cpu_config).unwrap();
    let coreml_session = net.create_session(coreml_config).unwrap();
    net.inputs(&cpu_session).iter().for_each(|x| {
        let mut tensor = x.tensor::<f32>().expect("No tensor");
        tensor.fill(1.0f32);
    });
    net.inputs(&coreml_session).iter().for_each(|x| {
        let mut tensor = x.tensor::<f32>().expect("No tensor");
        tensor.fill(1.0f32);
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
