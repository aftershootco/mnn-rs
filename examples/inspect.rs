use mnn::*;
use std::path::PathBuf;

#[derive(Debug, clap::Parser, Clone)]
pub struct Cli {
    model: PathBuf,
    #[arg(short, long)]
    forward: ForwardType,
    #[arg(short, long, default_value = "high")]
    power: PowerMode,
    #[arg(short = 'P', long, default_value = "high")]
    precision: PrecisionMode,
    #[arg(short, long, default_value = "high")]
    memory: MemoryMode,
    #[arg(short, long, default_value = "f32")]
    output_data_type: DataType,
    #[arg(short, long, default_value = "f32")]
    input_data_type: DataType,
    #[arg(short, long, default_value = "1")]
    loops: usize,
    #[arg(short, long)]
    no_cache: bool,
}

#[derive(Debug, Clone, clap::ValueEnum)]
pub enum DataType {
    F32,
    U8,
    I8,
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
    if !cli.no_cache {
        interpreter.set_cache_file(cli.model.with_extension("cache"), 128)?;
    }

    tracing_subscriber::fmt()
        .event_format(
            tracing_subscriber::fmt::format()
                .with_file(true)
                .with_line_number(true),
        )
        .init();

    let mut config = ScheduleConfig::new();
    config.set_type(cli.forward);
    let mut session = time!(interpreter.create_session(config)?; "create session");
    if !cli.no_cache {
        interpreter.update_cache_file(&mut session)?;
    }

    let mut current = 0;
    println!("--------------------------------Info--------------------------------");
    let mem = interpreter.memory(&session)?;
    let flops = interpreter.flops(&session)?;
    println!("Memory: {:?}MiB", mem);
    println!("Flops : {:?}M", flops);
    println!("ResizeStatus : {:?}", interpreter.resize_status(&session)?);

    time!(loop {
        println!("--------------------------------Inputs--------------------------------");
        interpreter.inputs(&session).iter().for_each(|x| {
            unsafe {
            match cli.input_data_type {
                DataType::F32 => {
                    let mut tensor = x.tensor_unresized::<f32>().expect("No tensor");
                    println!("{}: {:?}", x.name(), tensor.shape());
                    tensor.fill(1.0f32);
                },
                DataType::U8 => {
                    let mut tensor = x.tensor_unresized::<u8>().expect("No tensor");
                    println!("{}: {:?}", x.name(), tensor.shape());
                    tensor.fill(1u8);
                },
                DataType::I8 => {
                    let mut tensor = x.tensor_unresized::<i8>().expect("No tensor");
                    println!("{}: {:?}", x.name(), tensor.shape());
                    tensor.fill(1i8);
                },
            };
            }
        });

        println!("Running session");
        interpreter.run_session(&session)?;
        println!("--------------------------------Outputs--------------------------------");
        let outputs = interpreter.outputs(&session);
        outputs.iter().for_each(|x| {
            match cli.output_data_type {
                DataType::F32 => {
                    let tensor = x.tensor::<f32>().expect("No tensor");
                    println!("{}: {:?}", x.name(), tensor.shape());
                    time!(tensor.wait(MapType::MAP_TENSOR_READ, true); format!("Waiting for tensor: {}", x.name()));
                },
                DataType::U8 => {
                    let tensor = x.tensor::<u8>().expect("No tensor");
                    println!("{}: {:?}", x.name(), tensor.shape());
                    time!(tensor.wait(MapType::MAP_TENSOR_READ, true); format!("Waiting for tensor: {}", x.name()));
                },
                DataType::I8 => {
                    let tensor = x.tensor::<i8>().expect("No tensor");
                    println!("{}: {:?}", x.name(), tensor.shape());
                    time!(tensor.wait(MapType::MAP_TENSOR_READ, true); format!("Waiting for tensor: {}", x.name()));
                },
            };

        });
        current += 1;
        if current >= cli.loops {
            break;
        }
    }; "run loop");
    Ok(())
}
