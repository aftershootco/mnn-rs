use mnn::*;
use std::path::PathBuf;

#[derive(Debug, clap::Parser, Clone)]
pub struct Cli {
    model: PathBuf,
}
pub fn main() -> anyhow::Result<()> {
    use clap::Parser;
    let cli = Cli::parse();
    let mut interpreter = Interpreter::from_file(cli.model)?;

    let mut config = ScheduleConfig::new();
    config.set_type(ForwardType::CPU);
    let mut backend_config = BackendConfig::new();
    backend_config.set_precision_mode(PrecisionMode::High);
    backend_config.set_power_mode(PowerMode::High);

    config.set_backend_config(&backend_config);
    let now = std::time::Instant::now();
    let session = interpreter.create_session(&mut config)?;
    println!("create session time: {:?}", now.elapsed());
    // let inputs = interpreter.get_inputs(&session);
    // let outputs = interpreter.get_outputs(&session);
    // let mut image = inputs
    //     .iter()
    //     .find(|x| x.name() == "image")
    //     .expect("No input named image")
    //     .tensor();
    let mut image = interpreter.input(&session, "image")?;
    let mut image_tensor = image.create_host_tensor_from_device(false);
    let size = image_tensor.element_size();
    let unit_tensor_data = vec![1.0f32; size];
    image_tensor.print_shape();
    image_tensor.host_mut().copy_from_slice(&unit_tensor_data);

    let now = std::time::Instant::now();
    image.copy_from_host_tensor(&image_tensor)?;
    println!("copy time for image tensor: {:?}", now.elapsed());
    let output = interpreter.output::<f32>(&session, "dense")?;
    // mask.host_mut::<f32>().fill(0.7f32);

    // image.copy_from_host_tensor(&unit_tensor)?;

    // let mut looping = 0;
    // while looping < 10 {
    // looping += 1;
    let now = std::time::Instant::now();
    interpreter.run_session(&session)?;
    output.wait(ffi::MapType::MAP_TENSOR_READ, true);
    println!("run time: {:?}", now.elapsed());

    let now = std::time::Instant::now();
    let mut output_tensor = output.create_host_tensor_from_device(false);
    println!("create host tensor time: {:?}", now.elapsed());

    let now = std::time::Instant::now();
    output.copy_to_host_tensor(&mut output_tensor)?;
    println!("copy time: {:?}", now.elapsed());

    // let out_vec = output_tensor.host::<f32>().to_vec();
    // let mut out_ppm = b"P6\n512 512\n255\n".to_vec();
    // out_ppm.extend(out_vec.iter().map(|x| *x as u8));
    // std::fs::write(format!("output{looping}.ppm"), out_ppm)?;
    // }

    Ok(())
}
