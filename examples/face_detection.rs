// Face detection example using the UltraFace-320 model.
//
// Model: UltraFace RFB-320 (https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB)
//
// To get the model:
//   1. Download the ONNX model:
//      curl -L -o /tmp/version-RFB-320.onnx \
//        "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/ultraface/models/version-RFB-320.onnx"
//   2. Convert to MNN:
//      MNNConvert -f ONNX --modelFile /tmp/version-RFB-320.onnx --MNNModel examples/assets/ultraface-320.mnn
//
// In case model conversion fails, you can use onnxsim to simplify the ONNX model first before converting to MNN:
// https://github.com/onnxsim/onnxsim
//
// Usage:
//   cargo run --example face_detection -- <image> [--model PATH] [--threshold 0.7] [--iou-threshold 0.3]
use image::{DynamicImage, ImageDecoder, Rgb};
use imageproc::drawing::draw_hollow_rect_mut;
use imageproc::rect::Rect;
use mnn::*;
use std::path::PathBuf;

const INPUT_WIDTH: u32 = 320;
const INPUT_HEIGHT: u32 = 240;

#[derive(Debug, clap::Parser)]
struct Cli {
    /// Path to an input image (jpg, png, etc.)
    image: PathBuf,

    /// Path to the UltraFace MNN model
    #[arg(short, long, default_value = "examples/assets/ultraface-320.mnn")]
    model: PathBuf,

    /// Confidence threshold for face detections
    #[arg(short, long, default_value = "0.7")]
    threshold: f32,

    /// IoU threshold for non-maximum suppression
    #[arg(long, default_value = "0.3")]
    iou_threshold: f32,

    #[arg(short, long, default_value = "cpu")]
    forward: mnn::ForwardType,
}

#[derive(Debug, Clone)]
struct Face {
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    confidence: f32,
}

impl Face {
    /// Convert normalized coordinates back to pixel coordinates for a given image size
    fn to_pixel_coords(&self, img_width: u32, img_height: u32) -> (f32, f32, f32, f32) {
        (
            self.x1 * img_width as f32,
            self.y1 * img_height as f32,
            self.x2 * img_width as f32,
            self.y2 * img_height as f32,
        )
    }
}

impl std::fmt::Display for Face {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (px1, py1, px2, py2) = self.to_pixel_coords(INPUT_WIDTH, INPUT_HEIGHT);
        write!(
            f,
            "conf={:.1}% bbox=[{:.1}, {:.1}, {:.1}, {:.1}] size={:.0}x{:.0}",
            self.confidence * 100.0,
            px1,
            py1,
            px2,
            py2,
            px2 - px1,
            py2 - py1,
        )
    }
}

fn iou(a: &Face, b: &Face) -> f32 {
    let x1 = a.x1.max(b.x1);
    let y1 = a.y1.max(b.y1);
    let x2 = a.x2.min(b.x2);
    let y2 = a.y2.min(b.y2);
    let intersection = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
    let area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    let area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    let union = area_a + area_b - intersection;
    if union <= 0.0 {
        0.0
    } else {
        intersection / union
    }
}

fn nms(mut faces: Vec<Face>, iou_threshold: f32) -> Vec<Face> {
    faces.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
    let mut keep = Vec::new();
    let mut suppressed = vec![false; faces.len()];
    for i in 0..faces.len() {
        if suppressed[i] {
            continue;
        }
        keep.push(faces[i].clone());
        for j in (i + 1)..faces.len() {
            if !suppressed[j] && iou(&faces[i], &faces[j]) > iou_threshold {
                suppressed[j] = true;
            }
        }
    }
    keep
}

/// Preprocess image for UltraFace: resize to 320x240, RGB, normalize with mean=127 div=128.
/// Returns CHW float buffer in NCHW layout (batch dim handled by MNN).
fn preprocess_image(path: &std::path::Path) -> anyhow::Result<Vec<f32>> {
    let img = image::ImageReader::open(path)?;
    let img = img.with_guessed_format()?;
    let mut decoder = img.into_decoder()?;
    let o = decoder.orientation()?;
    let mut img = DynamicImage::from_decoder(decoder)?;
    img.apply_orientation(o);
    let img = img
        .resize_exact(
            INPUT_WIDTH,
            INPUT_HEIGHT,
            image::imageops::FilterType::Triangle,
        )
        .to_rgb8();

    let (w, h) = (img.width() as usize, img.height() as usize);
    let pixels = img.into_raw(); // HWC, RGB, u8

    // Convert to CHW float with normalization: (pixel - 127) / 128
    let mut chw = vec![0.0f32; 3 * h * w];
    for c in 0..3 {
        for y in 0..h {
            for x in 0..w {
                let src_idx = (y * w + x) * 3 + c;
                let dst_idx = c * h * w + y * w + x;
                chw[dst_idx] = (pixels[src_idx] as f32 - 127.0) / 128.0;
            }
        }
    }
    Ok(chw)
}

fn main() -> anyhow::Result<()> {
    use clap::Parser;
    let cli = Cli::parse();

    // Load and preprocess image
    let img_original = image::open(&cli.image)?;
    let (orig_w, orig_h) = (img_original.width(), img_original.height());
    println!("Image: {} ({}x{})", cli.image.display(), orig_w, orig_h);

    let input_data = preprocess_image(&cli.image)?;

    // Load model and create session
    let interpreter = Interpreter::from_file(&cli.model)?;
    let mut config = ScheduleConfig::new();
    config.set_type(cli.forward);
    let mut backend_config = BackendConfig::new();
    backend_config.set_precision_mode(PrecisionMode::High);
    config.set_backend_config(backend_config);
    let mut session = interpreter.create_session(config)?;

    let mem = interpreter.memory(&session)?;
    let flops = interpreter.flops(&session)?;
    println!(
        "Model: {} ({:.2} MiB, {:.2} MFLOPS)",
        cli.model.display(),
        mem,
        flops
    );

    // Copy preprocessed image into input tensor
    let input = interpreter.input::<f32>(&mut session, "input")?;
    println!("Input 'input': shape={:?}", input.shape());
    let mut host_tensor = input.create_host_tensor_from_device(false);
    host_tensor.host_mut().copy_from_slice(&input_data);
    input.copy_from_host_tensor(&host_tensor)?;

    // Run inference
    let now = std::time::Instant::now();
    interpreter.run_session(&session)?;
    println!("Inference time: {:?}", now.elapsed());

    // Read outputs
    let scores_tensor = interpreter.output::<f32>(&session, "scores")?;
    let boxes_tensor = interpreter.output::<f32>(&session, "boxes")?;
    let scores_host = scores_tensor.create_host_tensor_from_device(true);
    let boxes_host = boxes_tensor.create_host_tensor_from_device(true);
    let scores = scores_host.host();
    let boxes = boxes_host.host();

    // Parse detections: scores is [1, num_anchors, 2], boxes is [1, num_anchors, 4]
    let num_anchors = scores.len() / 2;
    let mut candidates = Vec::new();
    for i in 0..num_anchors {
        let face_score = scores[i * 2 + 1]; // index 1 = face class
        if face_score > cli.threshold {
            candidates.push(Face {
                x1: boxes[i * 4],
                y1: boxes[i * 4 + 1],
                x2: boxes[i * 4 + 2],
                y2: boxes[i * 4 + 3],
                confidence: face_score,
            });
        }
    }

    let detections = nms(candidates, cli.iou_threshold);
    println!(
        "\nDetected {} face(s) (threshold={}):",
        detections.len(),
        cli.threshold
    );
    for (i, face) in detections.iter().enumerate() {
        let (px1, py1, px2, py2) = face.to_pixel_coords(orig_w, orig_h);
        println!(
            "  [{}] {} (in original image: [{:.0}, {:.0}, {:.0}, {:.0}])",
            i, face, px1, py1, px2, py2
        );
    }

    if detections.is_empty() {
        println!("  (no faces detected)");
    }

    // Draw bounding boxes on the original image and save with _detected suffix
    if !detections.is_empty() {
        let mut canvas = img_original.to_rgb8();
        let green = Rgb([0u8, 255, 0]);

        for face in &detections {
            let (px1, py1, px2, py2) = face.to_pixel_coords(orig_w, orig_h);
            let x = px1.round() as i32;
            let y = py1.round() as i32;
            let w = (px2 - px1).round() as u32;
            let h = (py2 - py1).round() as u32;
            if w > 0 && h > 0 {
                draw_hollow_rect_mut(&mut canvas, Rect::at(x, y).of_size(w, h), green);
            }
        }

        let stem = cli.image.file_stem().unwrap_or_default().to_string_lossy();
        let ext = cli.image.extension().unwrap_or_default().to_string_lossy();
        let out_name = format!("{stem}_detected.{ext}");
        let out_path = cli.image.with_file_name(out_name);
        canvas.save(&out_path)?;
        println!("\nSaved result to {}", out_path.display());
    }

    Ok(())
}
