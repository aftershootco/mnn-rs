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
    #[clap(short, long, default_value = "f32")]
    output_data_type: DataType,
    #[clap(short, long, default_value = "f32")]
    input_data_type: DataType,
    #[clap(short, long, default_value = "1")]
    loops: usize,
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
    let net = Interpreter::from_file(&cli.model)?;
    net.set_cache_file(cli.model.with_extension("cache"), 128)?;

    let mut config = ScheduleConfig::new();
    config.set_type(cli.forward);
    let mut session = time!(net.create_session(config)?; "create session");
    net.update_cache_file(&mut session)?;

    let mut current = 0;
    println!("--------------------------------Info--------------------------------");
    let mem = net.memory(&session)?;
    let flops = net.flops(&session)?;
    println!("Memory: {:?}MiB", mem);
    println!("Flops : {:?}M", flops);
    println!("ResizeStatus : {:?}", net.resize_status(&session)?);

    time!(loop {
        println!("--------------------------------Inputs--------------------------------");
        net.inputs(&session).iter().for_each(|x| {
            match cli.input_data_type {
                DataType::F32 => {
                    let mut tensor = x.tensor::<f32>().expect("No tensor");
                    println!("{}: {:?}", x.name(), tensor.shape());
                    tensor.fill(1.0f32);
                },
                DataType::U8 => {
                    let mut tensor = x.tensor::<u8>().expect("No tensor");
                    println!("{}: {:?}", x.name(), tensor.shape());
                    tensor.fill(1u8);
                },
                DataType::I8 => {
                    let mut tensor = x.tensor::<i8>().expect("No tensor");
                    println!("{}: {:?}", x.name(), tensor.shape());
                    tensor.fill(1i8);
                },
            };
        });

        println!("Running session");
        net.run_session(&session)?;
        println!("--------------------------------Outputs--------------------------------");
        let outputs = net.outputs(&session);
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
