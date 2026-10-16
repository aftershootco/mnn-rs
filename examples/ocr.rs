// Full PaddleOCR (PP-OCRv5) pipeline example.
//
// Pipeline: text detection (DBNet) -> crop each text region -> text
// recognition (CRNN/SVTR) with CTC greedy decode -> print + draw results.
//
// Models (PaddleOCR PP-OCRv4, https://github.com/PaddlePaddle/PaddleOCR):
//   1. Download the ch PP-OCRv4 detection + recognition ONNX models (RapidOCR
//      mirror) and convert each to MNN:
//        curl -L -o det.onnx \
//          https://huggingface.co/SWHL/RapidOCR/resolve/main/PP-OCRv4/ch_PP-OCRv4_det_infer.onnx
//        curl -L -o rec.onnx \
//          https://huggingface.co/SWHL/RapidOCR/resolve/main/PP-OCRv4/ch_PP-OCRv4_rec_infer.onnx
//        MNNConvert -f ONNX --modelFile det.onnx \
//          --MNNModel examples/assets/paddle_det_v4.mnn
//        MNNConvert -f ONNX --modelFile rec.onnx \
//          --MNNModel examples/assets/paddle_rec_v4.mnn
//      If conversion fails, simplify the ONNX first with onnxsim
//      (https://github.com/onnxsim/onnxsim).
//   2. Fetch the character dictionary the recognition model was trained with.
//      This MUST match the model: the model's output class count equals the
//      number of dictionary lines + 1 (the CTC blank), or + 2 if the model also
//      appends a space class. The ch PP-OCRv4 rec model above uses:
//        curl -L -o examples/assets/ppocr_keys_v1.txt \
//          https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/ppocr/utils/ppocr_keys_v1.txt
//      Other charsets live under PaddleOCR/ppocr/utils/(dict/). The example
//      prints a warning if the dict size does not match the model.
//
// The det/rec inputs are dynamically shaped, so this example resizes the input
// tensor (resize_tensor_by_name + resize_session) before reading it.
//
// Tensor names vary by how the model was converted (MNNConvert prints them).
// Override via the CLI flags if your conversion differs from the defaults.
//
// Usage:
//   cargo run --example ocr -- <image>
use image::{DynamicImage, GrayImage, ImageDecoder, Luma, Rgb, RgbImage, imageops::FilterType};
use imageproc::contours::find_contours;
use imageproc::drawing::draw_line_segment_mut;
use imageproc::geometry::min_area_rect;
use mnn::*;
use std::path::PathBuf;

type Quad = [(f32, f32); 4];

#[derive(Debug, clap::Parser)]
struct Cli {
    /// Path to an input image (jpg, png, etc.)
    image: PathBuf,

    /// Path to the PP-OCR detection MNN model
    #[arg(long, default_value = "examples/assets/paddle_det_v4.mnn")]
    det_model: PathBuf,

    /// Path to the PP-OCR recognition MNN model
    #[arg(long, default_value = "examples/assets/paddle_rec_v4.mnn")]
    rec_model: PathBuf,

    /// Path to the character dictionary (e.g. ppocr_keys_v1.txt)
    #[arg(long, default_value = "examples/assets/ppocr_keys_v1.txt")]
    dict: PathBuf,

    #[arg(short, long, default_value = "cpu")]
    forward: mnn::ForwardType,

    /// Backend for the recognition stage (defaults to --forward)
    #[arg(long)]
    rec_forward: Option<mnn::ForwardType>,

    /// Binarization threshold for the detection probability map
    #[arg(long, default_value = "0.3")]
    det_thresh: f32,

    /// Minimum mean-probability score to keep a detected box
    #[arg(long, default_value = "0.5")]
    box_thresh: f32,

    /// How much to expand (unclip) each detected box
    #[arg(long, default_value = "1.5")]
    unclip_ratio: f32,

    /// Max side length the detection input is resized to (rounded to /32)
    #[arg(long, default_value = "960")]
    det_limit: u32,

    /// Recognition input height
    #[arg(long, default_value = "48")]
    rec_height: u32,

    /// Max recognition input width. Wide lines are squashed to this width, so a
    /// small value (PP-OCR trains at 320) garbles long lines. 0 = auto: derive
    /// the ceiling from the image resolution so no line is ever squashed.
    #[arg(long, default_value = "0")]
    rec_max_width: u32,

    #[arg(long, default_value = "x")]
    det_input: String,
    #[arg(long, default_value = "sigmoid_0.tmp_0")]
    det_output: String,
    #[arg(long, default_value = "x")]
    rec_input: String,
    #[arg(long, default_value = "softmax_11.tmp_0")]
    rec_output: String,
}

/// Load an image applying EXIF orientation.
fn load_image(path: &std::path::Path) -> anyhow::Result<RgbImage> {
    let reader = image::ImageReader::open(path)?.with_guessed_format()?;
    let mut decoder = reader.into_decoder()?;
    let orientation = decoder.orientation()?;
    let mut img = DynamicImage::from_decoder(decoder)?;
    img.apply_orientation(orientation);
    Ok(img.to_rgb8())
}

/// Round up to the nearest multiple of 32 (minimum 32).
fn round32(v: u32) -> u32 {
    v.div_ceil(32).max(1) * 32
}

/// Detection preprocess: resize so the longest side <= `limit` and both sides
/// are multiples of 32, normalize with ImageNet mean/std, return NCHW buffer.
/// Returns (chw, resized_w, resized_h, ratio_w, ratio_h).
fn det_preprocess(img: &RgbImage, limit: u32) -> (Vec<f32>, u32, u32, f32, f32) {
    let (w, h) = (img.width(), img.height());
    let max_side = w.max(h) as f32;
    let scale = if max_side > limit as f32 {
        limit as f32 / max_side
    } else {
        1.0
    };
    let wd = round32((w as f32 * scale).round() as u32);
    let hd = round32((h as f32 * scale).round() as u32);
    let resized = image::imageops::resize(img, wd, hd, FilterType::Triangle);

    let ratio_w = wd as f32 / w as f32;
    let ratio_h = hd as f32 / h as f32;

    let mean = [0.485f32, 0.456, 0.406];
    let std = [0.229f32, 0.224, 0.225];
    let (ww, hh) = (wd as usize, hd as usize);
    let mut chw = vec![0f32; 3 * hh * ww];
    for c in 0..3 {
        for y in 0..hh {
            for x in 0..ww {
                let px = resized.get_pixel(x as u32, y as u32)[c] as f32;
                chw[c * hh * ww + y * ww + x] = (px / 255.0 - mean[c]) / std[c];
            }
        }
    }
    (chw, wd, hd, ratio_w, ratio_h)
}

fn polygon_area(q: &Quad) -> f32 {
    let mut a = 0.0;
    for i in 0..4 {
        let (x1, y1) = q[i];
        let (x2, y2) = q[(i + 1) % 4];
        a += x1 * y2 - x2 * y1;
    }
    (a / 2.0).abs()
}

fn polygon_perimeter(q: &Quad) -> f32 {
    let mut p = 0.0;
    for i in 0..4 {
        let (x1, y1) = q[i];
        let (x2, y2) = q[(i + 1) % 4];
        p += ((x2 - x1).powi(2) + (y2 - y1).powi(2)).sqrt();
    }
    p
}

/// Expand a box outward by `distance = area * ratio / perimeter`, approximating
/// the Vatti/clipper "unclip" PaddleOCR applies after detection. Each corner is
/// offset along its two adjacent edges, which shifts every side out by the same
/// `distance` regardless of the box aspect ratio (a plain centroid-push would
/// under-expand the short side of wide text lines). Corners must be ordered
/// around the polygon, as `min_area_rect` returns them.
fn unclip(q: &Quad, ratio: f32) -> Quad {
    let area = polygon_area(q);
    let perim = polygon_perimeter(q).max(1e-6);
    let distance = area * ratio / perim;
    let unit = |a: (f32, f32), b: (f32, f32)| {
        let (dx, dy) = (b.0 - a.0, b.1 - a.1);
        let len = (dx * dx + dy * dy).sqrt().max(1e-6);
        (dx / len, dy / len)
    };
    let mut out = *q;
    for i in 0..4 {
        let cur = q[i];
        let to_next = unit(cur, q[(i + 1) % 4]);
        let to_prev = unit(cur, q[(i + 3) % 4]);
        out[i].0 = cur.0 - (to_next.0 + to_prev.0) * distance;
        out[i].1 = cur.1 - (to_next.1 + to_prev.1) * distance;
    }
    out
}

/// Mean probability over the axis-aligned bounding box of `q` (coords in the
/// detection-resized space, matching `prob`).
fn box_score(prob: &[f32], w: u32, h: u32, q: &Quad) -> f32 {
    let min_x = q
        .iter()
        .map(|p| p.0)
        .fold(f32::MAX, f32::min)
        .floor()
        .max(0.0) as u32;
    let max_x = q
        .iter()
        .map(|p| p.0)
        .fold(f32::MIN, f32::max)
        .ceil()
        .min((w - 1) as f32) as u32;
    let min_y = q
        .iter()
        .map(|p| p.1)
        .fold(f32::MAX, f32::min)
        .floor()
        .max(0.0) as u32;
    let max_y = q
        .iter()
        .map(|p| p.1)
        .fold(f32::MIN, f32::max)
        .ceil()
        .min((h - 1) as f32) as u32;
    if max_x <= min_x || max_y <= min_y {
        return 0.0;
    }
    let mut sum = 0.0;
    let mut n = 0u32;
    for y in min_y..=max_y {
        for x in min_x..=max_x {
            sum += prob[(y * w + x) as usize];
            n += 1;
        }
    }
    if n == 0 { 0.0 } else { sum / n as f32 }
}

fn quad_dims(q: &Quad) -> (f32, f32) {
    let dist = |a: (f32, f32), b: (f32, f32)| ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt();
    let w = dist(q[0], q[1]).max(dist(q[2], q[3]));
    let h = dist(q[1], q[2]).max(dist(q[0], q[3]));
    (w, h)
}

/// DBNet post-process: probability map [1,1,H,W] -> quad boxes in ORIGINAL
/// image coordinates.
#[allow(clippy::too_many_arguments)]
fn db_postprocess(
    prob: &[f32],
    w: u32,
    h: u32,
    orig_w: f32,
    orig_h: f32,
    ratio_w: f32,
    ratio_h: f32,
    cli: &Cli,
) -> Vec<Quad> {
    let mut bitmap = GrayImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let v = if prob[(y * w + x) as usize] >= cli.det_thresh {
                255
            } else {
                0
            };
            bitmap.put_pixel(x, y, Luma([v]));
        }
    }

    let mut boxes = Vec::new();
    for contour in find_contours::<u32>(&bitmap) {
        if contour.points.len() < 4 {
            continue;
        }
        let rect = min_area_rect(&contour.points);
        let q: Quad = rect.map(|p| (p.x as f32, p.y as f32));

        if box_score(prob, w, h, &q) < cli.box_thresh {
            continue;
        }
        let (bw, bh) = quad_dims(&q);
        if bw.min(bh) < 3.0 {
            continue;
        }

        let expanded = unclip(&q, cli.unclip_ratio);
        let mapped = expanded.map(|(x, y)| {
            (
                (x / ratio_w).clamp(0.0, orig_w),
                (y / ratio_h).clamp(0.0, orig_h),
            )
        });
        boxes.push(mapped);
    }
    boxes
}

/// Crop a detected text region into an upright image. Uses the axis-aligned
/// bounding box of the quad (pragmatic: a true perspective warp would be more
/// correct for rotated text but adds significant code). Rotates 90 deg for
/// tall (vertical) crops.
fn get_rotate_crop(img: &RgbImage, q: &Quad) -> RgbImage {
    let min_x = q
        .iter()
        .map(|p| p.0)
        .fold(f32::MAX, f32::min)
        .floor()
        .max(0.0) as u32;
    let max_x = q
        .iter()
        .map(|p| p.0)
        .fold(f32::MIN, f32::max)
        .ceil()
        .min(img.width() as f32) as u32;
    let min_y = q
        .iter()
        .map(|p| p.1)
        .fold(f32::MAX, f32::min)
        .floor()
        .max(0.0) as u32;
    let max_y = q
        .iter()
        .map(|p| p.1)
        .fold(f32::MIN, f32::max)
        .ceil()
        .min(img.height() as f32) as u32;
    let cw = max_x.saturating_sub(min_x).max(1);
    let ch = max_y.saturating_sub(min_y).max(1);

    let crop = image::imageops::crop_imm(img, min_x, min_y, cw, ch).to_image();
    if ch as f32 > cw as f32 * 1.5 {
        image::imageops::rotate90(&crop)
    } else {
        crop
    }
}

/// Recognition preprocess: resize to fixed height keeping aspect ratio,
/// normalize to [-1, 1], return (NCHW buffer, target_width).
fn rec_preprocess(crop: &RgbImage, h: u32, max_w: u32) -> (Vec<f32>, u32) {
    let (cw, ch) = (crop.width(), crop.height());
    let wt = ((h as f32 * cw as f32 / ch as f32).ceil() as u32).clamp(1, max_w);
    let resized = image::imageops::resize(crop, wt, h, FilterType::Triangle);

    let (ww, hh) = (wt as usize, h as usize);
    let mut chw = vec![0f32; 3 * hh * ww];
    for c in 0..3 {
        for y in 0..hh {
            for x in 0..ww {
                let px = resized.get_pixel(x as u32, y as u32)[c] as f32;
                chw[c * hh * ww + y * ww + x] = (px / 255.0 - 0.5) / 0.5;
            }
        }
    }
    (chw, wt)
}

/// CTC greedy decode. Output logits are [1, T, C] (already softmaxed). Index 0
/// is the CTC blank; class index `i` maps to `dict[i-1]`. PaddleOCR appends a
/// space as the last class, so `i-1 == dict.len()` decodes to a space.
fn ctc_decode(logits: &[f32], shape: &[i32], dict: &[String]) -> (String, f32) {
    let t = shape[1] as usize;
    let cls = shape[2] as usize;
    let mut text = String::new();
    let mut prev = usize::MAX;
    let mut conf_sum = 0.0;
    let mut conf_n = 0u32;
    for ti in 0..t {
        let base = ti * cls;
        let mut best = 0usize;
        let mut best_v = logits[base];
        for ci in 1..cls {
            let v = logits[base + ci];
            if v > best_v {
                best_v = v;
                best = ci;
            }
        }
        if best != 0 && best != prev {
            let ci = best - 1;
            if ci < dict.len() {
                text.push_str(&dict[ci]);
            } else if ci == dict.len() {
                text.push(' ');
            }
            conf_sum += best_v;
            conf_n += 1;
        }
        prev = best;
    }
    let conf = if conf_n > 0 {
        conf_sum / conf_n as f32
    } else {
        0.0
    };
    (text, conf)
}

fn load_dict(path: &std::path::Path) -> anyhow::Result<Vec<String>> {
    let content = std::fs::read_to_string(path)?;
    let mut dict: Vec<String> = content
        .split('\n')
        .map(|l| l.trim_end_matches('\r').to_string())
        .collect();
    if dict.last().is_some_and(|l| l.is_empty()) {
        dict.pop();
    }
    Ok(dict)
}

fn build_session(interpreter: &Interpreter, forward: ForwardType) -> anyhow::Result<Session<'_>> {
    let mut config = ScheduleConfig::new();
    config.set_type(forward);
    let mut backend_config = BackendConfig::new();
    backend_config.set_precision_mode(PrecisionMode::High);
    config.set_backend_config(backend_config);
    Ok(interpreter.create_session(config)?)
}

fn main() -> anyhow::Result<()> {
    use clap::Parser;
    let cli = Cli::parse();

    let dict = load_dict(&cli.dict)?;
    let image = load_image(&cli.image)?;
    let (orig_w, orig_h) = (image.width(), image.height());
    println!(
        "Image: {} ({}x{}), dict: {} chars",
        cli.image.display(),
        orig_w,
        orig_h,
        dict.len()
    );

    // --- Detection ---
    let det = Interpreter::from_file(&cli.det_model)?;
    let mut det_session = build_session(&det, cli.forward)?;

    let (det_input_data, wd, hd, ratio_w, ratio_h) = det_preprocess(&image, cli.det_limit);
    det.resize_tensor_by_name(
        &mut det_session,
        &cli.det_input,
        [1, 3, hd as i32, wd as i32],
    )?;
    det.resize_session(&mut det_session);

    let input = det.input::<f32>(&mut det_session, &cli.det_input)?;
    let mut host = input.create_host_tensor_from_device(false);
    host.host_mut().copy_from_slice(&det_input_data);
    input.copy_from_host_tensor(&host)?;

    let now = std::time::Instant::now();
    det.run_session(&det_session)?;
    println!(
        "Detection input {}x{}, inference {:?}",
        wd,
        hd,
        now.elapsed()
    );

    let det_out = det.output::<f32>(&det_session, &cli.det_output)?;
    let det_out_host = det_out.create_host_tensor_from_device(true);
    let prob = det_out_host.host();
    let boxes = db_postprocess(
        prob,
        wd,
        hd,
        orig_w as f32,
        orig_h as f32,
        ratio_w,
        ratio_h,
        &cli,
    );
    println!("Detected {} text region(s)", boxes.len());

    // --- Recognition ---
    // Auto width ceiling: a cropped line can be at most the image's longest side
    // wide, so rec_height * max(w, h) bounds the aspect-preserving width without
    // ever squashing a real line (boxes thinner than 3px are already dropped).
    let rec_max_width = if cli.rec_max_width == 0 {
        cli.rec_height * orig_w.max(orig_h)
    } else {
        cli.rec_max_width
    };

    let rec = Interpreter::from_file(&cli.rec_model)?;
    let mut rec_session = build_session(&rec, cli.rec_forward.unwrap_or(cli.forward))?;

    let mut results: Vec<(Quad, String, f32)> = Vec::new();
    let mut checked_dict = false;
    for q in &boxes {
        let crop = get_rotate_crop(&image, q);
        let (rec_data, wt) = rec_preprocess(&crop, cli.rec_height, rec_max_width);

        rec.resize_tensor_by_name(
            &mut rec_session,
            &cli.rec_input,
            [1, 3, cli.rec_height as i32, wt as i32],
        )?;
        rec.resize_session(&mut rec_session);

        let rin = rec.input::<f32>(&mut rec_session, &cli.rec_input)?;
        let mut rhost = rin.create_host_tensor_from_device(false);
        rhost.host_mut().copy_from_slice(&rec_data);
        rin.copy_from_host_tensor(&rhost)?;

        rec.run_session(&rec_session)?;

        let rout = rec.output::<f32>(&rec_session, &cli.rec_output)?;
        let rout_host = rout.create_host_tensor_from_device(true);
        let shape = rout.shape();

        // The dictionary must match the model: number of output classes equals
        // dict lines + 1 (CTC blank), or + 2 if the model also appends a space.
        // A mismatch yields high-confidence gibberish, so warn loudly.
        if !checked_dict {
            checked_dict = true;
            let classes = shape.as_ref()[2] as usize;
            if classes != dict.len() + 1 && classes != dict.len() + 2 {
                eprintln!(
                    "warning: model has {classes} output classes but dict has {} entries \
                     (expected {} or {}). Recognized text will be wrong — use the dictionary \
                     that matches this model.",
                    dict.len(),
                    classes - 1,
                    classes - 2,
                );
            }
        }

        let (text, conf) = ctc_decode(rout_host.host(), shape.as_ref(), &dict);
        if !text.is_empty() {
            results.push((*q, text, conf));
        }
    }

    println!("\nRecognized {} line(s):", results.len());
    for (i, (q, text, conf)) in results.iter().enumerate() {
        println!(
            "  [{}] conf={:.1}% box=[({:.0},{:.0}),({:.0},{:.0}),({:.0},{:.0}),({:.0},{:.0})] \"{}\"",
            i,
            conf * 100.0,
            q[0].0,
            q[0].1,
            q[1].0,
            q[1].1,
            q[2].0,
            q[2].1,
            q[3].0,
            q[3].1,
            text
        );
    }

    // Draw boxes on the original image and save with an _ocr suffix.
    if !results.is_empty() {
        let mut canvas = image.clone();
        let green = Rgb([0u8, 255, 0]);
        for (q, _, _) in &results {
            for k in 0..4 {
                draw_line_segment_mut(&mut canvas, q[k], q[(k + 1) % 4], green);
            }
        }
        let stem = cli.image.file_stem().unwrap_or_default().to_string_lossy();
        let ext = cli.image.extension().unwrap_or_default().to_string_lossy();
        let out_path = cli.image.with_file_name(format!("{stem}_ocr.{ext}"));
        canvas.save(&out_path)?;
        println!("\nSaved result to {}", out_path.display());
    }

    Ok(())
}
