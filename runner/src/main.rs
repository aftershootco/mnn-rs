use anyhow::Result;
// use clap::Parser;
use mnn::*;
use wasm_bindgen::prelude::*;

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

// #[derive(Parser)]
// pub struct Cli {
//     #[clap(short, long)]
//     pub model: PathBuf,
//     #[clap(short, long, default_value = "cpu")]
//     forward: mnn::utils::ForwardType,
//     #[clap(short, long, default_value = "high")]
//     pub precision: mnn::utils::Modes,
//     #[clap(short = 'P', long, default_value = "high")]
//     pub power: mnn::utils::Modes,
//     #[clap(
//         short,
//         long,
//         default_value_t = std::thread::available_parallelism().expect("No available threads")
//     )]
//     pub threads: std::num::NonZeroUsize,
// }
//
// impl Cli {
//     pub fn out_name(&self, out: impl AsRef<str>) -> Result<PathBuf> {
//         let current_dir = std::env::current_dir()?;
//         let model_name = self
//             .model
//             .file_stem()
//             .and_then(|x| x.to_str().map(|x| x.to_string()))
//             .ok_or_else(|| {
//                 anyhow::anyhow!("Could not get file name from path: {:?}", self.model)
//             })?;
//         Ok(current_dir.join(format!(
//             "{}_{}_{:?}_Precision{:?}_Power{:?}",
//             out.as_ref(),
//             model_name,
//             self.forward,
//             self.precision,
//             self.power
//         )))
//     }
// }

#[wasm_bindgen(main)]
fn main() {
    // main_err().expect("Failed to run")
}

#[wasm_bindgen]
pub fn test_bench(model: js_sys::Uint8Array) -> Result<(), JsError> {
    main_err(model).map_err(|e| JsError::new(&format!("{:?}", e)))
}

fn main_err(model: js_sys::Uint8Array) -> Result<()> {
    dbg!("sus");
    println!("Called test_bench");

    // let cli = Cli::parse();
    // let mut net = mnn::Interpreter::from_file("sus")?;
    let mut net = mnn::Interpreter::from_bytes(model.to_vec())?;
    //     let mut config = ScheduleConfig::new();
    //     // config.set_type(cli.forward.to_forward_type());
    //     config.set_num_threads(1);
    //     let backend_config = BackendConfig::new();
    //     // backend_config.set_precision_mode(cli.precision.to_precision_mode());
    //     // backend_config.set_power_mode(cli.power.to_power_mode());
    //     config.set_backend_config(&backend_config);
    //
    //     net.set_session_mode(SessionMode::Session_Release);
    //     let session = time!(net.create_session(&mut config)?; "Loading model");
    //     let inputs = net.inputs(&session);
    //
    //     for input in inputs.iter() {
    //         let name = input.name();
    //         let mut tensor = input.tensor::<f32>()?;
    //         let mut cpu_tensor = tensor.create_host_tensor_from_device(false);
    //         tensor.print_shape();
    //         cpu_tensor.print_shape();
    //         time!(cpu_tensor.host_mut().fill(1.0); format!("Filling tensor {}", name));
    //         time!(tensor.copy_from_host_tensor(&cpu_tensor)?; format!("Copying tensor {}", name));
    //     }
    //
    //     time!(net.run_session(&session)?; "Running session");
    //
    //     let outputs = net.outputs(&session);
    //     for output in outputs.iter() {
    //         let name = output.name();
    //         let tensor = output.tensor::<f32>()?;
    //         time!(tensor.wait(mnn::ffi::MapType::MAP_TENSOR_READ, true); format!("Waiting tensor {}", name));
    //
    //         let cpu_tensor = time!(tensor.create_host_tensor_from_device(true);
    //          format!("Creating and Copying to host tensor {}", name));
    //         cpu_tensor.print_shape();
    //         let shape = cpu_tensor.shape();
    //         let n = cpu_tensor.batch();
    //         let c = cpu_tensor.channel();
    //         let h = cpu_tensor.height();
    //         let w = cpu_tensor.width();
    //         // match (n, c, h, w) {
    //         //     (1, 3, _, _) if h == w && h != 0 => {
    //         //         println!("Saving output tensor {} as image", name);
    //         //         let out_vec = cpu_tensor.host().to_vec();
    //         //         let mut out_ppm: Vec<u8> = format!("P6\n{w} {h}\n255\n").bytes().collect();
    //         //         // let mut out_ppm = b"P6\n512 512\n255\n".to_vec();
    //         //         out_ppm.extend(out_vec.iter().map(|x: &f32| *x as u8));
    //         //         std::fs::write(cli.out_name(name)?.with_extension("ppm"), out_ppm)?;
    //         //     }
    //         //     // (128 | 16, 3 | 2, _, _) => {
    //         //     _ if shape.size == 2 => {
    //         //         let json = serde_json::to_string_pretty(&cpu_tensor.host())?;
    //         //         println!("Saving output tensor {}.json as json: ", name);
    //         //         // println!("{}", json);
    //         //         std::fs::write(cli.out_name(name)?.with_extension("json"), json)?;
    //         //     }
    //         //     _ => {
    //         //         println!("Saving output tensor {} as binary", name);
    //         //         let data = cpu_tensor.host();
    //         //         std::fs::write(
    //         //             cli.out_name(name)?.with_extension("bin"),
    //         //             bytemuck::cast_slice(data),
    //         //         )?;
    //         //     }
    //         // }
    //     }
    //
    Ok(())
}
