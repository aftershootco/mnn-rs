use mnn_sys::SessionMode;

const NO_BRAINER: &[u8] = include_bytes!("../models/nobrainerf16.mnn");
macro_rules! time {
    ($($x:expr),+ ; $text: expr) => {
        {
            let start = std::time::Instant::now();
            let result = { $($x);+ };
            let elapsed = start.elapsed();
            println!("{}: took: {:?}", $text,elapsed );
            result
        }
    };
    ($($x:expr),+) => {
        time!($($x),+; "")
    };
}
pub extern "C" fn mnn_benchmark(forward: i32) {
    let mut net = crate::Interpreter::from_bytes(NO_BRAINER).unwrap();
    let mut config = crate::ScheduleConfig::new();
    config.set_type(match forward {
        1 => mnn_sys::MNNForwardType::MNN_FORWARD_OPENCL,
        _ => mnn_sys::MNNForwardType::MNN_FORWARD_CPU,
    });
    let mut backend_config = crate::BackendConfig::new();
    backend_config.set_precision_mode(mnn_sys::PrecisionMode::Precision_Normal);
    backend_config.set_power_mode(mnn_sys::PowerMode::Power_High);
    config.set_backend_config(&backend_config);
    let session = net.create_session(&mut config).unwrap();
    net.set_session_mode(SessionMode::Session_Release);

    let inputs = net.inputs(&session);

    for input in inputs.iter() {
        let name = input.name();
        let mut tensor = input.tensor::<f32>().unwrap();
        let mut cpu_tensor = tensor.create_host_tensor_from_device(false);
        tensor.print_shape();
        cpu_tensor.print_shape();
        cpu_tensor.host_mut().fill(1.0);
        tensor.copy_from_host_tensor(&cpu_tensor).unwrap();
    }
    time!(net.run_session(&session).unwrap(); "Running session");
    let outputs = net.outputs(&session);
    for output in outputs.iter() {
        let name = output.name();
        let tensor = output.tensor::<f32>().unwrap();
        tensor.wait(mnn_sys::MapType::MAP_TENSOR_READ, true);
        println!("Output tensor name: {}", name);
        let cpu_tensor = tensor.create_host_tensor_from_device(true);
        cpu_tensor.print_shape();
    }
}
