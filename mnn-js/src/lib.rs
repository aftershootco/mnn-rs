use js_sys::Uint8Array;
use wasm_bindgen::prelude::*;
use mnn::*;

#[wasm_bindgen]
pub fn bench(model: Uint8Array) {
    let model = model.to_vec();
    let mut net = Interpreter::from_bytes(&model).unwrap();
    let mut config = ScheduleConfig::new();
    // config.set_type()
    config.set_num_threads(1);
    let session = net.create_session(&mut config).unwrap();
    let inputs = net.inputs(&session);
    for input in inputs.iter() {
        let tensor = input.tensor::<f32>().unwrap();
        println!("Filling tensor {} with 1.0", input.name());
        let mut cpu_tensor = tensor.create_host_tensor_from_device(false);
        cpu_tensor.host_mut().fill(1.0);
    }

    net.run_session(&session).unwrap();
}
