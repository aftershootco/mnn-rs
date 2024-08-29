use anyhow::Result;
use candice::*;
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
    forward: mnn::ForwardType,
    // #[clap(short, long, default_value = "high")]
    // pub precision: mnn::PrecisionMode,
    // #[clap(short = 'P', long, default_value = "high")]
    // pub power: mnn::PowerMode,
    #[clap(
        short,
        long,
        default_value_t = std::thread::available_parallelism().expect("No available threads")
    )]
    pub threads: std::num::NonZeroUsize,
    #[clap(short, long)]
    pub resize_batch: Option<u16>,
    #[clap(short, long)]
    pub input: Option<PathBuf>,
}

impl Cli {
    pub fn out_name(&self, out: impl AsRef<str>) -> Result<PathBuf> {
        let current_dir = std::env::current_dir()?;
        let model_name = self
            .model
            .file_stem()
            .and_then(|x| x.to_str().map(|x| x.to_string()))
            .ok_or_else(|| {
                anyhow::anyhow!("Could not get file name from path: {:?}", self.model)
            })?;
        Ok(current_dir.join(format!(
            // "{}_{}_{:?}_Precision{:?}_Power{:?}",
            "{}_{}_{:?}",
            out.as_ref(),
            model_name,
            self.forward,
            // self.precision,
            // self.power
        )))
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let mut net = mnn::Interpreter::from_file(&cli.model)?;
    let mut config = ScheduleConfig::new();
    config.set_type(cli.forward);
    config.set_num_threads(cli.threads.get() as i32);
    let mut backend_config = BackendConfig::new();
    backend_config.set_precision_mode(PrecisionMode::High);
    backend_config.set_power_mode(PowerMode::High);
    config.set_backend_config(&backend_config);

    net.set_session_mode(SessionMode::Release);
    let mut session = time!(net.create_session(&mut config)?; "Loading model".red());
    let inputs = net.inputs(&session);
    let outputs = net.outputs(&session);

    if let Some(resize_batch) = cli.resize_batch {
        for input in inputs.iter() {
            let mut tensor = input.tensor::<f32>()?;
            let mut shape = tensor.shape().as_ref().to_owned();
            shape[0] = resize_batch as i32;
            net.resize_tensor(&mut tensor, shape);
        }
        // for output in outputs.iter() {
        //     let mut tensor = output.tensor::<f32>()?;
        //     let mut shape = tensor.shape().as_ref().to_owned();
        //     shape[0] = resize_batch as i32;
        //     net.resize_tensor(&mut tensor, shape);
        // }
        net.resize_session(&mut session);
    }

    for input in inputs.iter() {
        let name = input.name();
        let mut tensor = input.tensor::<f32>()?;
        let mut cpu_tensor = tensor.create_host_tensor_from_device(false);
        if let Some(input) = &cli.input {
            let image = std::fs::read(input)?;
            let image_tj = turbojpeg::decompress(&image, turbojpeg::PixelFormat::RGB)?;
            let image = image_tj.pixels;
            let input_image = fast_image_resize::images::Image::from_vec_u8(
                image_tj.width as u32,
                image_tj.height as u32,
                image,
                fast_image_resize::PixelType::U8x3,
            )?;
            let shape = cpu_tensor.shape();
            let shape = shape.as_ref();
            let x = shape[1] as u32;
            let y = shape[0] as u32;
            let mut output_image =
                fast_image_resize::images::Image::new(x, y, fast_image_resize::PixelType::U8x3);
            let mut resizer = fast_image_resize::Resizer::new();
            resizer.resize(&input_image, &mut output_image, None)?;
            let out_float: Vec<f32> = output_image.buffer().iter().map(|f| *f as f32).collect();

            // let out_float: &[f32] = time!(bytemuck::cast_slice(&image));
            cpu_tensor.host_mut().copy_from_slice(&out_float);
        } else {
            time!(cpu_tensor.host_mut().fill(1.0); format!("Filling tensor {}", name.green()));
        }
        let cpu_input = cpu_tensor.host();
        std::fs::write("input_bin", bytemuck::cast_slice(cpu_input))?;
        time!(tensor.copy_from_host_tensor(&cpu_tensor)?; format!("Copying tensor {}", name.yellow()));
    }

    time!(net.run_session(&session)?; "Running session".blue());

    let outputs = net.outputs(&session);
    for output in outputs.iter() {
        let name = output.name();
        let tensor = output.tensor()?;
        time!(tensor.wait(mnn::ffi::MapType::MAP_TENSOR_READ, true); format!("Waiting tensor {}", name.red()));

        let cpu_tensor = time!(tensor.create_host_tensor_from_device(true);
         format!("Creating and Copying to host tensor {}", name.green()));
        cpu_tensor.print_shape();
        let shape = cpu_tensor.shape();
        let n = cpu_tensor.batch();
        let c = cpu_tensor.channel();
        let w = cpu_tensor.width();
        let h = cpu_tensor.height();
        match (n, c, w, h) {
            (1, 3, _, _) if h == w && h != 0 => {
                println!("Saving output tensor {} as image", name.green());
                let out_vec = cpu_tensor.host().to_vec();
                let mut out_ppm: Vec<u8> = format!("P6\n{w} {h}\n255\n").bytes().collect();
                out_ppm.extend(out_vec.iter().map(|x: &f32| *x as u8));
                std::fs::write(cli.out_name(name)?.with_extension("ppm"), out_ppm)?;
            }
            // (128 | 16, 3 | 2, _, _) => {
            _ if shape.size == 2 => {
                let json = serde_json::to_string_pretty(&cpu_tensor.host())?;
                println!("Saving output tensor {}.json as json: ", name.green());
                // println!("{}", json);
                std::fs::write(cli.out_name(name)?.with_extension("json"), json)?;
            }
            _ => {
                println!("Saving output tensor {} as binary", name.blue());
                let data = cpu_tensor.host();
                std::fs::write(
                    cli.out_name(name)?.push_extension("bin"),
                    bytemuck::cast_slice(data),
                )?;
            }
        }
    }

    Ok(())
}
