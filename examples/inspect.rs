use mnn::*;
use std::path::PathBuf;

#[derive(Debug, clap::Parser, Clone)]
pub struct Cli {
    model: PathBuf,
    #[clap(short, long)]
    forward: ForwardType,
    #[clap(short, long, default_value = "high")]
    power: PowerMode,
    #[clap(short = 'P', long, default_value = "high")]
    precision: PrecisionMode,
    #[clap(short, long, default_value = "high")]
    memory: MemoryMode,
    #[clap(short, long, default_value = "1")]
    loops: usize,
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
    let mut interpreter = Interpreter::from_file(&cli.model)?;
    interpreter.set_cache_file(cli.model.with_extension("cache"), 128)?;

    let mut config = ScheduleConfig::new();
    config.set_type(cli.forward);
    let mut session = time!(interpreter.create_session(config)?; "create session");
    interpreter.update_cache_file(&mut session)?;

    let mut current = 0;
    time!(loop {
        {
            let inputs = interpreter.inputs(&session);
            inputs.iter().for_each(|x| {
                let mut tensor = x.tensor::<f32>().expect("No tensor");
                println!("{}: {:?}", x.name(), tensor.shape());
                tensor.fill(1.0f32);
            });
        }
        time!(interpreter.run_session_with_callback(&session, |tensors, name| {
            println!("Before Callback: {:?}", name);
            1
        },|_ , name| {
            println!("After Callback: {:?}", name);
            1
        } , true)?;"run session");
        let outputs = interpreter.outputs(&session);
        outputs.iter().for_each(|x| {
            let tensor = x.tensor::<f32>().expect("No tensor");
            time!(tensor.wait(ffi::MapType::MAP_TENSOR_READ, true); format!("Waiting for tensor: {}", x.name()));
            println!("{}: {:?}", x.name(), tensor.shape());
            let _ = tensor.create_host_tensor_from_device(true);
            // std::fs::write(format!("{}.bin", x.name()), bytemuck::cast_slice(cpu_tensor.host())).expect("Unable to write");
        });
        current += 1;
        if current >= cli.loops {
            break;
        }
    }; "run loop");
    Ok(())
}
