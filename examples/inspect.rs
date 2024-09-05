use mnn::*;
use std::path::PathBuf;

#[derive(Debug, clap::Parser, Clone)]
pub struct Cli {
    model: PathBuf,
    #[clap(short, long)]
    forward: ForwardType,
    #[clap(short, long, default_value = "high")]
    power: PowerMode,
    #[clap(short, long, default_value = "high")]
    precision: PrecisionMode,
}

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

pub fn main() -> anyhow::Result<()> {
    use clap::Parser;
    let cli = Cli::parse();
    let mut interpreter = Interpreter::from_file(cli.model)?;

    let mut config = ScheduleConfig::new();
    config.set_type(cli.forward);
    let mut backend_config = BackendConfig::new();
    backend_config.set_precision_mode(PrecisionMode::High);
    backend_config.set_power_mode(PowerMode::High);

    config.set_backend_config(backend_config);
    // let handle = mnn::sync::SessionHandle::new(interpreter, config)?;
    let session = time!(interpreter.create_session(config)?; "create session");
    // handle.run(|sr| {
    //     let interpreter = sr.interpreter();
    //     let session = sr.session();
    let inputs = interpreter.inputs(&session);
    inputs.iter().for_each(|x| {
        let mut tensor = x.tensor::<f32>().expect("No tensor");
        println!("{}: {:?}", x.name(), tensor.shape());
        tensor.fill(1.0f32);
        // let mut cpu_tensor = tensor.create_host_tensor_from_device(false);
        // cpu_tensor.host_mut().fill(1.0f32);
        // tensor
        //     .copy_from_host_tensor(&cpu_tensor)
        //     .expect("Could not copy tensor");
    });
    time!(interpreter.run_session(&session)?;"run session");
    let outputs = interpreter.outputs(&session);
    outputs.iter().for_each(|x| {
        let tensor = x.tensor::<f32>().expect("No tensor");
        time!(tensor.wait(ffi::MapType::MAP_TENSOR_READ, true); format!("Waiting for tensor: {}", x.name()));
        println!("{}: {:?}", x.name(), tensor.shape());
        let _ = tensor.create_host_tensor_from_device(true);
        // std::fs::write(format!("{}.bin", x.name()), bytemuck::cast_slice(cpu_tensor.host())).expect("Unable to write");
    });
    // Ok(())
    // })?;
    Ok(())
}
