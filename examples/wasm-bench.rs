use mnn_sys::SessionMode;

fn main() {}
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

#[no_mangle]
pub extern "C" fn drop_boxed_slice(ptr: *mut *const f32) {
    unsafe {
        Box::from_raw(ptr);
    }
}

// #[no_mangle]
// pub extern "C" fn len_boxed_slice(ptr: *mut *const f32) -> i32{
//     unsafe {
//         let b = Box::<[f32]>::from_raw(ptr.cast::<&[f32]>());
//         let len = b.len();
//         core::mem::forget(b);
//         return len as i32;
//     }
// }

#[no_mangle]
pub extern "C" fn mnn_benchmark(forward: i32) -> *mut *const f32 {
    let mut net = mnn::Interpreter::from_bytes(NO_BRAINER).unwrap();
    let mut config = mnn::ScheduleConfig::new();
    config.set_type(match forward {
        1 => mnn_sys::MNNForwardType::MNN_FORWARD_OPENCL,
        _ => mnn_sys::MNNForwardType::MNN_FORWARD_CPU,
    });
    config.set_num_threads(1);
    let mut backend_config = mnn::BackendConfig::new();
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
    let output = outputs.iter().next().expect("Failed to get outpu");
    let name = output.name();
    let tensor = output.tensor::<f32>().unwrap();
    tensor.wait(mnn_sys::MapType::MAP_TENSOR_READ, true);
    println!("Output tensor name: {}", name);
    let cpu_tensor = tensor.create_host_tensor_from_device(true);
    cpu_tensor.print_shape();

    let ret = Box::into_raw(cpu_tensor.host().to_vec().into_boxed_slice());
    ret.cast()
}
