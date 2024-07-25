use ffi::{BackendConfig, ScheduleConfig};
use mnn::*;
use std::path::PathBuf;

#[derive(Debug, clap::Parser, Clone)]
pub struct Cli {
    image: PathBuf,
    model: PathBuf,
}
pub fn main() -> anyhow::Result<()> {
    use clap::Parser;
    let cli = Cli::parse();
    let mut interpreter = Interpreter::from_file(cli.model)?;

    let mut config = ScheduleConfig::new();
    config.type_ = ffi::MNNForwardType::MNN_FORWARD_CPU;
    let mut backend_config = BackendConfig::new();
    backend_config.precision = ffi::PrecisionMode::Precision_High;
    backend_config.power = ffi::PowerMode::Power_High;

    config.backendConfig = core::ptr::addr_of!(backend_config).cast_mut();
    let session = interpreter.create_session(&config)?;
    let inputs = interpreter.get_inputs(&session);
    let outputs = interpreter.get_outputs(&session);

   // let img = zune_image::image::Image::open(&cli.image)?;
   // let img: Vec<f32> = resize(img, 512, 512)?.into_iter().map(|x| x as f32).collect();
    let img: Vec<f32> = std::fs::read(&cli.image)?.into_iter().map(|x| x as f32).collect();

    let mut image = inputs
        .iter()
        .find(|x| x.name() == "image")
        .expect("No input named image")
        .tensor();
    let mut mask = inputs
        .iter()
        .find(|x| x.name() == "mask")
        .expect("No input named mask")
        .tensor();
    image.host_mut::<f32>().copy_from_slice(&img);
    // let unit_tensor_data = vec![1.0f32; 1 * 3 * 512 * 512];
    // let mut unit_tensor = image.create_host_tensor_from_device(false);
    // unit_tensor.host_mut().copy_from_slice(&unit_tensor_data);
    mask.host_mut::<f32>().fill(0.7f32);

    // image.copy_from_host_tensor(&unit_tensor)?;

    interpreter.run_session(&session)?;
    let output = outputs
        .iter()
        .find(|x| x.name() == "output")
        .expect("Not output named output")
        .tensor();
    let out_vec = output.host::<f32>().to_vec();
    let mut out_ppm = b"P6\n512 512\n255\n".to_vec();
    out_ppm.extend(out_vec.iter().map(|x| *x as u8));
    std::fs::write("output.ppm", out_ppm)?;

    Ok(())
}
use anyhow::Result;
use zune_core::colorspace::ColorSpace;

pub fn zune_to_fir(
    image: zune_image::image::Image,
) -> Result<fast_image_resize::images::Image<'static>> {
    let colorspace = image.colorspace();
    let pixel_type = match colorspace {
        ColorSpace::RGB => fast_image_resize::PixelType::U8x3,
        ColorSpace::RGBA => fast_image_resize::PixelType::U8x4,
        _ => return Err(anyhow::anyhow!("Unsupported colorspace: {colorspace:?}",)),
    };
    let (width, height) = image.dimensions();
    let mut image = image.flatten_to_u8();
    let image = image.pop().expect("Failed to get the image frame");
    let image =
        fast_image_resize::images::Image::from_vec_u8(width as _, height as _, image, pixel_type)?;
    Ok(image)
}

pub fn fir_to_zune(image: fast_image_resize::images::Image) -> Result<zune_image::image::Image> {
    let colorspace = match image.pixel_type() {
        // fast_image_resize::PixelType::U8 => 1,
        // fast_image_resize::PixelType::U8x2 => 2,
        fast_image_resize::PixelType::U8x3 => ColorSpace::RGB,
        fast_image_resize::PixelType::U8x4 => ColorSpace::RGBA,
        _ => {
            return Err(anyhow::anyhow!(
                "Unsupported number of components: {:?}",
                image.pixel_type()
            ))
        }
    };
    let image = zune_image::image::Image::from_u8(
        image.buffer(),
        image.width() as _,
        image.height() as _,
        colorspace,
    );
    Ok(image)
}

pub fn resize(input: zune_image::image::Image, width: u32, height: u32) -> Result<Vec<u8>> {
    use fast_image_resize::images::Image;
    use fast_image_resize::Resizer;

    let img = zune_to_fir(input)?;
    let mut dst = Image::new(width, height, img.pixel_type());
    let mut resizer = Resizer::new();
    resizer.resize(&img, &mut dst, None)?;

    Ok(dst.into_vec())
}
