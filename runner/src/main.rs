use anyhow::Result;
use clap::Parser;
use mnn::*;
use owo_colors::OwoColorize;
use std::path::PathBuf;

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

#[derive(Parser)]
pub struct Cli {
    #[clap(short, long)]
    pub model: PathBuf,
    #[clap(short, long, default_value = "cpu")]
    forward: mnn::utils::ForwardType,
    #[clap(short, long, default_value = "high")]
    pub precision: mnn::utils::Modes,
    #[clap(short = 'P', long, default_value = "high")]
    pub power: mnn::utils::Modes,
    #[clap(
        short,
        long,
        default_value_t = std::thread::available_parallelism().expect("No available threads")
    )]
    pub threads: std::num::NonZeroUsize,
}

impl Cli {
    pub fn out_name(&self, out: impl AsRef<str>) -> Result<String> {
        let model_name = self
            .model
            .file_stem()
            .and_then(|x| x.to_str().map(|x| x.to_string()))
            .ok_or_else(|| {
                anyhow::anyhow!("Could not get file name from path: {:?}", self.model)
            })?;
        Ok(format!(
            "{}_{}_{:?}_Precision::{:?}_Power::{:?}",
            out.as_ref(),
            model_name,
            self.forward,
            self.precision,
            self.power
        ))
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let mut net = mnn::Interpreter::from_file(&cli.model)?;
    let mut config = ScheduleConfig::new();
    config.set_type(cli.forward.to_forward_type());
    config.set_num_threads(cli.threads.get() as i32);
    let mut backend_config = BackendConfig::new();
    backend_config.set_precision_mode(cli.precision.to_precision_mode());
    backend_config.set_power_mode(cli.power.to_power_mode());
    config.set_backend_config(&backend_config);

    let session = time!(net.create_session(&mut config)?; "Loading model".red());
    let inputs = net.inputs(&session);

    for input in inputs.iter() {
        let name = input.name();
        let mut tensor = input.tensor();
        let mut cpu_tensor = tensor.create_host_tensor_from_device(false);
        tensor.print_shape();
        cpu_tensor.print_shape();
        tensor.wait(mnn::ffi::MapType::MAP_TENSOR_WRITE, true);
        cpu_tensor.wait(mnn::ffi::MapType::MAP_TENSOR_WRITE, true);
        time!(cpu_tensor.host_mut::<f32>().fill(1.0f32); format!("Filling tensor {}", name.green()));
        time!(tensor.copy_from_host_tensor(&cpu_tensor)?; format!("Copying tensor {}", name.yellow()));
    }

    time!(net.run_session(&session)?; "Running session".blue());

    let outputs = net.outputs(&session);
    for output in outputs.iter() {
        let name = output.name();
        let tensor = output.tensor();
        time!(tensor.wait(mnn::ffi::MapType::MAP_TENSOR_READ, true); format!("Waiting tensor {}", name.red()));

        let cpu_tensor = time!(tensor.create_host_tensor_from_device(true);
         format!("Creating and Copying to host tensor {}", name.green()));
        cpu_tensor.print_shape();
        // dbg!(cpu_tensor.get_type());
        // dbg!(cpu_tensor.get_diemension_type());
        let shape = cpu_tensor.shape();
        let n = cpu_tensor.batch();
        let c = cpu_tensor.channel();
        let h = cpu_tensor.height();
        let w = cpu_tensor.width();
        match (n, c, h, w) {
            (1, 3, _, _) if h == w && h != 0 => {
                println!("Saving output tensor {} as image", name.green());
                let out_vec = cpu_tensor.host::<f32>().to_vec();
                let mut out_ppm: Vec<u8> = format!("P6\n{w} {h}\n255\n").bytes().collect();
                // let mut out_ppm = b"P6\n512 512\n255\n".to_vec();
                out_ppm.extend(out_vec.iter().map(|x| *x as u8));
                std::fs::write(format!("{}.bin", cli.out_name(name)?), out_ppm)?;
            }
            // (128 | 16, 3 | 2, _, _) => {
            _ if shape.size == 2 => {
                let json = serde_json::to_string_pretty(&cpu_tensor.host::<f32>())?;
                println!("Saving output tensor {}.json as json: ", name.green());
                // println!("{}", json);
                std::fs::write(format!("{}.json", cli.out_name(name)?), json)?;
            }
            _ => {
                println!("Saving output tensor {} as binary", name.blue());
                let data = cpu_tensor.host::<f32>();
                std::fs::write(
                    format!("{}.bin", cli.out_name(name)?),
                    bytemuck::cast_slice(data),
                )?;
            }
        }
    }

    Ok(())
}