use ::tap::*;
use anyhow::*;
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use std::{
    fs::Permissions,
    path::{Path, PathBuf},
    sync::LazyLock,
};
const VENDOR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/vendor");
const MANIFEST_DIR: &str = env!("CARGO_MANIFEST_DIR");
static TARGET_OS: LazyLock<String> =
    LazyLock::new(|| std::env::var("CARGO_CFG_TARGET_OS").expect("CARGO_CFG_TARGET_OS not set"));
static TARGET_ARCH: LazyLock<String> = LazyLock::new(|| {
    std::env::var("CARGO_CFG_TARGET_ARCH").expect("CARGO_CFG_TARGET_ARCH not found")
});
static EMSCRIPTEN_CACHE: LazyLock<String> = LazyLock::new(|| {
    let emscripten_cache = std::process::Command::new("em-config")
        .arg("CACHE")
        .output()
        .expect("Failed to get emscripten cache")
        .stdout;
    let emscripten_cache = std::str::from_utf8(&emscripten_cache)
        .expect("Failed to parse emscripten cache")
        .trim()
        .to_string();
    emscripten_cache
});

fn ensure_vendor_exists(vendor: impl AsRef<Path>) -> Result<()> {
    if vendor
        .as_ref()
        .read_dir()
        .with_context(|| format!("Vendor directory missing: {}", vendor.as_ref().display()))?
        .flatten()
        .count()
        == 0
    {
        anyhow::bail!("Vendor not found maybe you need to run \"git submodule update --init\"")
    }
    Ok(())
}

fn main() -> Result<()> {
    println!("cargo:rerun-if-changed=build.rs");
    let out_dir = PathBuf::from(std::env::var("OUT_DIR")?);
    let source = PathBuf::from(
        std::env::var("MNN_SRC")
            .ok()
            .unwrap_or_else(|| VENDOR.into()),
    );

    ensure_vendor_exists(&source)?;

    let vendor = out_dir.join("vendor");
    if !vendor.exists() {
        fs_extra::dir::copy(
            &source,
            &vendor,
            &fs_extra::dir::CopyOptions::new()
                .overwrite(true)
                .copy_inside(true),
        )
        .context("Failed to copy vendor")?;
        let intptr = vendor.join("include").join("MNN").join("Interpreter.hpp");
        #[cfg(unix)]
        std::fs::set_permissions(&intptr, Permissions::from_mode(0o644))?;
        try_patch_file("patches/typedef_template.patch", &intptr)
            .context("Failed to patch vendor")?;
        let intptr = vendor.join("include").join("MNN").join("HalideRuntime.h");
        #[cfg(unix)]
        std::fs::set_permissions(&intptr, Permissions::from_mode(0o644))?;
        try_patch_file("patches/halide_type_t_64.patch", intptr)
            .context("Failed to patch vendor")?;
    }

    let install_dir = out_dir.join("mnn-install");
    build_cmake(&vendor, &install_dir)?;

    mnn_c_build(PathBuf::from(MANIFEST_DIR).join("mnn_c"), &vendor)
        .with_context(|| "Failed to build mnn_c")?;
    mnn_c_bindgen(&vendor, &out_dir).with_context(|| "Failed to generate mnn_c bindings")?;
    println!("cargo:include={vendor}/include", vendor = vendor.display());
    if *TARGET_OS == "macos" {
        #[cfg(feature = "metal")]
        println!("cargo:rustc-link-lib=framework=Foundation");
        #[cfg(feature = "metal")]
        println!("cargo:rustc-link-lib=framework=CoreGraphics");
        #[cfg(feature = "metal")]
        println!("cargo:rustc-link-lib=framework=Metal");
        #[cfg(feature = "coreml")]
        println!("cargo:rustc-link-lib=framework=CoreML");
        #[cfg(feature = "coreml")]
        println!("cargo:rustc-link-lib=framework=CoreVideo");
        #[cfg(feature = "opencl")]
        println!("cargo:rustc-link-lib=framework=OpenCL");
        #[cfg(feature = "opengl")]
        println!("cargo:rustc-link-lib=framework=OpenGL");
    } else {
        // #[cfg(feature = "opencl")]
        // println!("cargo:rustc-link-lib=static=opencl");
    }
    if is_emscripten() {
        // println!("cargo:rustc-link-lib=static=stdc++");
        let emscripten_cache = std::process::Command::new("em-config")
            .arg("CACHE")
            .output()?
            .stdout;
        let emscripten_cache = std::str::from_utf8(&emscripten_cache)?.trim();
        let wasm32_emscripten_libs =
            PathBuf::from(emscripten_cache).join("sysroot/lib/wasm32-emscripten");
        println!(
            "cargo:rustc-link-search=native={}",
            wasm32_emscripten_libs.display()
        );
    }
    println!(
        "cargo:rustc-link-search=native={}",
        install_dir.join("lib").display()
    );
    println!("cargo:rustc-link-lib=static=MNN");
    Ok(())
}

pub fn mnn_c_bindgen(vendor: impl AsRef<Path>, out: impl AsRef<Path>) -> Result<()> {
    let vendor = vendor.as_ref();
    let mnn_c = PathBuf::from(MANIFEST_DIR).join("mnn_c");
    mnn_c.read_dir()?.flatten().for_each(|e| {
        rerun_if_changed(e.path());
    });
    const HEADERS: &[&str] = &[
        "error_code_c.h",
        "interpreter_c.h",
        "tensor_c.h",
        "backend_c.h",
        "schedule_c.h",
    ];

    let bindings = bindgen::Builder::default()
        // .clang_args(["-x", "c++"])
        .clang_arg(CxxOption::VULKAN.cxx())
        .clang_arg(CxxOption::METAL.cxx())
        .clang_arg(CxxOption::COREML.cxx())
        .clang_arg(CxxOption::OPENCL.cxx())
        .pipe(|builder| {
            if is_emscripten() {
                println!("cargo:rustc-cdylib-link-arg=-fvisibility=default");
                builder
                    .clang_arg("-fvisibility=default")
                    .clang_arg("--target=wasm32-emscripten")
                    .clang_arg(format!("-I{}/sysroot/include", emscripten_cache()))
            } else {
                builder
            }
        })
        .clang_arg(format!("-I{}", vendor.join("include").to_string_lossy()))
        .pipe(|generator| {
            HEADERS.iter().fold(generator, |gen, header| {
                gen.header(mnn_c.join(header).to_string_lossy())
            })
        })
        .newtype_enum("MemoryMode")
        .newtype_enum("PowerMode")
        .newtype_enum("PrecisionMode")
        .constified_enum_module("SessionMode")
        .rustified_enum("DimensionType")
        .rustified_enum("HandleDataType")
        .rustified_enum("MapType")
        .rustified_enum("halide_type_code_t")
        .rustified_enum("ErrorCode")
        .rustified_enum("MNNGpuMode")
        .rustified_enum("MNNForwardType")
        .rustified_enum("RuntimeStatus")
        .no_copy("CString")
        .generate_cstr(true)
        .generate_inline_functions(true)
        .size_t_is_usize(true)
        .emit_diagnostics()
        .detect_include_paths(std::env::var("TARGET") == std::env::var("HOST"))
        .ctypes_prefix("core::ffi")
        // .tap(|d| {
        //     // eprintln!("Full bindgen: {}", d.command_line_flags().join(" "));
        //     std::fs::write("bindgen.txt", d.command_line_flags().join(" ")).ok();
        // })
        .generate()?;
    bindings.write_to_file(out.as_ref().join("mnn_c.rs"))?;
    Ok(())
}

pub fn mnn_c_build(path: impl AsRef<Path>, vendor: impl AsRef<Path>) -> Result<()> {
    let mnn_c = path.as_ref();
    let files = mnn_c.read_dir()?.flatten().map(|e| e.path()).filter(|e| {
        e.extension() == Some(std::ffi::OsStr::new("cpp"))
            || e.extension() == Some(std::ffi::OsStr::new("c"))
    });
    let vendor = vendor.as_ref();
    cc::Build::new()
        .include(vendor.join("include"))
        // .includes(vulkan_includes(vendor))
        .pipe(|config| {
            #[cfg(feature = "vulkan")]
            config.define("MNN_VULKAN", "1");
            #[cfg(feature = "opengl")]
            config.define("MNN_OPENGL", "1");
            #[cfg(feature = "metal")]
            config.define("MNN_METAL", "1");
            #[cfg(feature = "coreml")]
            config.define("MNN_COREML", "1");
            #[cfg(feature = "opencl")]
            config.define("MNN_OPENCL", "ON");
            if is_emscripten() {
                config.compiler("emcc");
                // We can't compile wasm32-unknown-unknown with emscripten
                config.target("wasm32-unknown-emscripten");
                config.cpp_link_stdlib("c++-noexcept");
            }
            #[cfg(feature = "crt_static")]
            config.static_crt(true);
            config
        })
        .cpp(true)
        .static_flag(true)
        .files(files)
        .std("c++14")
        // .pipe(|build| {
        //     let c = build.get_compiler();
        //     use std::io::Write;
        //     writeln!(
        //         std::fs::File::create("./command.txt").unwrap(),
        //         "{:?}",
        //         c.to_command()
        //     )
        //     .unwrap();
        //     build
        // })
        .try_compile("mnn_c")
        .context("Failed to compile mnn_c library")?;
    Ok(())
}

pub fn build_cc(path: impl AsRef<Path>) -> Result<()> {
    let core = [
        "core/AutoTime.cpp",
        "core/Backend.cpp",
        "core/BufferAllocator.cpp",
        "core/ConvolutionCommon.cpp",
        "core/Execution.cpp",
        "core/FileLoader.cpp",
        "core/Interpreter.cpp",
        "core/MNNFileUtils.cpp",
        "core/MNNMemoryUtils.cpp",
        "core/OpCommonUtils.cpp",
        "core/Pipeline.cpp",
        "core/RuntimeFactory.cpp",
        "core/Schedule.cpp",
        "core/Session.cpp",
        "core/Tensor.cpp",
        "core/TensorUtils.cpp",
        "core/WrapExecution.cpp",
    ];
    let cv = [
        "cv/Matrix_CV.cpp",
        "cv/ImageProcessUtils.cpp",
        "cv/ImageProcess.cpp",
    ];
    let shape = [
        "shape/ShapeArgMax.cpp",
        "shape/ShapeAttention.cpp",
        "shape/ShapeBatchToSpaceND.cpp",
        "shape/ShapeBinaryOp.cpp",
        "shape/ShapeBroadcastTo.cpp",
        "shape/ShapeCast.cpp",
        "shape/ShapeConcat.cpp",
        "shape/ShapeConvTranspose3D.cpp",
        "shape/ShapeConvolution.cpp",
        "shape/ShapeConvolution3D.cpp",
        "shape/ShapeCosineSimilarity.cpp",
        "shape/ShapeCrop.cpp",
        "shape/ShapeCropAndResize.cpp",
        "shape/ShapeDeconvolution.cpp",
        "shape/ShapeDepthToSpace.cpp",
        "shape/ShapeDequantize.cpp",
        "shape/ShapeDet.cpp",
        "shape/ShapeDetectionOutput.cpp",
        "shape/ShapeDetectionPostProcess.cpp",
        "shape/ShapeDynamicQuant.cpp",
        "shape/ShapeExpandDims.cpp",
        "shape/ShapeFill.cpp",
        "shape/ShapeGatherND.cpp",
        "shape/ShapeGatherV2.cpp",
        "shape/ShapeGridSample.cpp",
        "shape/ShapeHistogram.cpp",
        "shape/ShapeInnerProduct.cpp",
        "shape/ShapeInterp.cpp",
        "shape/ShapeLSTM.cpp",
        "shape/ShapeLinSpace.cpp",
        "shape/ShapeMatMul.cpp",
        "shape/ShapeMoments.cpp",
        "shape/ShapeNonMaxSuppressionV2.cpp",
        "shape/ShapeOneHot.cpp",
        "shape/ShapePack.cpp",
        "shape/ShapePadding.cpp",
        "shape/ShapePermute.cpp",
        "shape/ShapePlugin.cpp",
        "shape/ShapePool.cpp",
        "shape/ShapePool3D.cpp",
        "shape/ShapePriorbox.cpp",
        "shape/ShapeProposal.cpp",
        "shape/ShapeQuantizedAvgPool.cpp",
        "shape/ShapeQuantizedMaxPool.cpp",
        "shape/ShapeRNNSequenceGRU.cpp",
        "shape/ShapeROIAlign.cpp",
        "shape/ShapeROIPooling.cpp",
        "shape/ShapeRandomUniform.cpp",
        "shape/ShapeRange.cpp",
        "shape/ShapeReduction.cpp",
        "shape/ShapeRegister.cpp",
        "shape/ShapeReshape.cpp",
        "shape/ShapeResize.cpp",
        "shape/ShapeScatterNd.cpp",
        "shape/ShapeSegmentMean.cpp",
        "shape/ShapeSelect.cpp",
        "shape/ShapeSetDiff1D.cpp",
        "shape/ShapeShape.cpp",
        "shape/ShapeSize.cpp",
        "shape/ShapeSlice.cpp",
        "shape/ShapeSliceTf.cpp",
        "shape/ShapeSpaceToBatchND.cpp",
        "shape/ShapeSpaceToDepth.cpp",
        "shape/ShapeSplitGelu.cpp",
        "shape/ShapeSqueeze.cpp",
        "shape/ShapeStridedSlice.cpp",
        "shape/ShapeSvd.cpp",
        "shape/ShapeTensorArray.cpp",
        "shape/ShapeTensorConvert.cpp",
        "shape/ShapeTile.cpp",
        "shape/ShapeTopKV2.cpp",
        "shape/ShapeTranspose.cpp",
        "shape/ShapeUnique.cpp",
        "shape/ShapeUnpack.cpp",
        "shape/ShapeUnravelIndex.cpp",
        "shape/ShapeWhere.cpp",
        "shape/SizeComputer.cpp",
    ];
    let utils = ["utils/InitNet.cpp", "utils/JNIHelper.cpp"];
    let geometry = [
        "geometry/ConvertUtils.cpp",
        "geometry/GeometryBatchMatMul.cpp",
        "geometry/GeometryBinary.cpp",
        "geometry/GeometryBroadcastTo.cpp",
        "geometry/GeometryComputer.cpp",
        "geometry/GeometryComputerUtils.cpp",
        "geometry/GeometryConcat.cpp",
        "geometry/GeometryConv2D.cpp",
        "geometry/GeometryConv2DBackPropFilter.cpp",
        "geometry/GeometryConv3D.cpp",
        "geometry/GeometryConvUtils.cpp",
        "geometry/GeometryConvert.cpp",
        "geometry/GeometryCosineSimilarity.cpp",
        "geometry/GeometryCrop.cpp",
        "geometry/GeometryCumSum.cpp",
        "geometry/GeometryDepthToSpace.cpp",
        "geometry/GeometryDet.cpp",
        "geometry/GeometryDilation2D.cpp",
        "geometry/GeometryELU.cpp",
        "geometry/GeometryFill.cpp",
        "geometry/GeometryGather.cpp",
        "geometry/GeometryImageOp.cpp",
        "geometry/GeometryInnerProduct.cpp",
        "geometry/GeometryLRN.cpp",
        "geometry/GeometryLSTM.cpp",
        "geometry/GeometryLayernorm.cpp",
        "geometry/GeometryOPRegister.cpp",
        "geometry/GeometryPermute.cpp",
        "geometry/GeometryPoolGrad.cpp",
        "geometry/GeometryPooling3D.cpp",
        "geometry/GeometryReduce.cpp",
        "geometry/GeometryReshape.cpp",
        "geometry/GeometryReverseSequence.cpp",
        "geometry/GeometryScatter.cpp",
        "geometry/GeometrySelect.cpp",
        "geometry/GeometryShape.cpp",
        "geometry/GeometrySlice.cpp",
        "geometry/GeometrySpaceToBatchND.cpp",
        "geometry/GeometrySpatialProduct.cpp",
        "geometry/GeometryStridedSlice.cpp",
        "geometry/GeometryTensorArray.cpp",
        "geometry/GeometryThreshold.cpp",
        "geometry/GeometryTile.cpp",
        "geometry/GeometryTopK.cpp",
        "geometry/GeometryUnary.cpp",
    ];

    let files = cc::Build::new().files(core).compile("mnn_cc");
    Ok(())
}

pub fn build_cmake(path: impl AsRef<Path>, install: impl AsRef<Path>) -> Result<()> {
    let threads = std::thread::available_parallelism()?;
    cmake::Config::new(path)
        .cxxflag("-std=c++14")
        .parallel(threads.get() as u8)
        .define("MNN_BUILD_SHARED_LIBS", "OFF")
        .define("MNN_SEP_BUILD", "OFF")
        .define("MNN_PORTABLE_BUILD", "ON")
        .define("MNN_USE_SYSTEM_LIB", "OFF")
        .define("MNN_BUILD_CONVERTER", "OFF")
        .define("MNN_BUILD_TOOLS", "OFF")
        .define("CMAKE_INSTALL_PREFIX", install.as_ref())
        // https://github.com/rust-lang/rust/issues/39016
        // https://github.com/rust-lang/cc-rs/pull/717
        // .define("CMAKE_BUILD_TYPE", "Release")
        .pipe(|config| {
            config.define("MNN_WIN_RUNTIME_MT", CxxOption::CRT_STATIC.cmake_value());
            config.define("MNN_USE_THREAD_POOL", CxxOption::THREADPOOL.cmake_value());
            config.define("MNN_OPENMP", CxxOption::OPENMP.cmake_value());
            config.define("MNN_VULKAN", CxxOption::VULKAN.cmake_value());
            config.define("MNN_METAL", CxxOption::METAL.cmake_value());
            config.define("MNN_COREML", CxxOption::COREML.cmake_value());
            config.define("MNN_OPENCL", CxxOption::OPENCL.cmake_value());
            config.define("MNN_OPENGL", CxxOption::OPENGL.cmake_value());
            // #[cfg(windows)]
            if *TARGET_OS == "windows" {
                config.define("CMAKE_CXX_FLAGS", "-DWIN32=1");
            }

            if is_emscripten() {
                config
                    .define("CMAKE_C_COMPILER", "emcc")
                    .define("CMAKE_CXX_COMPILER", "em++")
                    .target("wasm32-unknown-emscripten");
            }
            config
        })
        .build();
    Ok(())
}

pub fn try_patch_file(patch: impl AsRef<Path>, file: impl AsRef<Path>) -> Result<()> {
    let patch = dunce::canonicalize(patch)?;
    rerun_if_changed(&patch);
    let patch = std::fs::read_to_string(&patch)?;
    let patch = diffy::Patch::from_str(&patch)?;
    let file_path = file.as_ref();
    let file = std::fs::read_to_string(file_path).context("Failed to read input file")?;
    let patched_file =
        diffy::apply(&file, &patch).context("Failed to apply patches using diffy")?;
    std::fs::write(file_path, patched_file)?;
    Ok(())
}

pub fn rerun_if_changed(path: impl AsRef<Path>) {
    println!("cargo:rerun-if-changed={}", path.as_ref().display());
}

// pub fn vulkan_includes(vendor: impl AsRef<Path>) -> Vec<PathBuf> {
//     let vendor = vendor.as_ref();
//     let vulkan_dir = vendor.join("source/backend/vulkan");
//     if cfg!(feature = "vulkan") {
//         vec![
//             vulkan_dir.clone(),
//             vulkan_dir.join("runtime"),
//             vulkan_dir.join("component"),
//             // IDK If the order is important but the cmake file does it like this
//             vulkan_dir.join("buffer/execution"),
//             vulkan_dir.join("buffer/backend"),
//             vulkan_dir.join("buffer"),
//             vulkan_dir.join("buffer/shaders"),
//             // vulkan_dir.join("image/execution"),
//             // vulkan_dir.join("image/backend"),
//             // vulkan_dir.join("image"),
//             // vulkan_dir.join("image/shaders"),
//             vendor.join("schema/current"),
//             vendor.join("3rd_party/flatbuffers/include"),
//             vendor.join("source"),
//         ]
//     } else {
//         vec![]
//     }
// }

pub fn is_emscripten() -> bool {
    *TARGET_OS == "emscripten" && *TARGET_ARCH == "wasm32"
}

pub fn emscripten_cache() -> &'static str {
    &EMSCRIPTEN_CACHE
}

#[derive(Debug, Clone, Copy)]
pub enum CxxOptionValue {
    On,
    Off,
    Value(&'static str),
}

impl From<bool> for CxxOptionValue {
    fn from(b: bool) -> Self {
        if b {
            Self::On
        } else {
            Self::Off
        }
    }
}

impl CxxOptionValue {
    pub const fn from_bool(value: bool) -> Self {
        match value {
            true => Self::On,
            false => Self::Off,
        }
    }
}

impl From<&'static str> for CxxOptionValue {
    fn from(s: &'static str) -> Self {
        match s {
            "ON" => Self::On,
            "OFF" => Self::Off,
            "1" => Self::On,
            "0" => Self::Off,
            "true" => Self::On,
            "false" => Self::Off,
            _ => Self::Value(s),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CxxOption {
    pub name: &'static str,
    pub value: CxxOptionValue,
}

macro_rules! cxx_option_from_feature {
    ($feature:literal, $cxx:literal) => {{
        CxxOption::from_bool($cxx, cfg!(feature = $feature))
    }};
}
impl CxxOption {
    const fn from_bool(name: &'static str, value: bool) -> Self {
        Self {
            name,
            value: CxxOptionValue::from_bool(value),
        }
    }
    pub const VULKAN: CxxOption = cxx_option_from_feature!("vulkan", "MNN_VULKAN");
    pub const METAL: CxxOption = cxx_option_from_feature!("metal", "MNN_METAL");
    pub const COREML: CxxOption = cxx_option_from_feature!("coreml", "MNN_COREML");
    pub const OPENCL: CxxOption = cxx_option_from_feature!("opencl", "MNN_OPENCL");
    pub const OPENMP: CxxOption = cxx_option_from_feature!("openmp", "MNN_OPENMP");
    pub const OPENGL: CxxOption = cxx_option_from_feature!("opengl", "MNN_OPENGL");
    pub const CRT_STATIC: CxxOption = cxx_option_from_feature!("opengl", "MNN_WIN_RUNTIME_MT");
    pub const THREADPOOL: CxxOption =
        cxx_option_from_feature!("mnn-threadpool", "MNN_USE_THREAD_POOL");

    pub fn new(name: &'static str, value: impl Into<CxxOptionValue>) -> Self {
        Self {
            name,
            value: value.into(),
        }
    }

    pub fn on(mut self) -> Self {
        self.value = CxxOptionValue::On;
        self
    }

    pub fn off(mut self) -> Self {
        self.value = CxxOptionValue::Off;
        self
    }

    pub fn with_value(mut self, value: &'static str) -> Self {
        self.value = CxxOptionValue::Value(value);
        self
    }

    pub fn cmake(&self) -> String {
        match &self.value {
            CxxOptionValue::On => format!("-D{}=ON", self.name),
            CxxOptionValue::Off => format!("-D{}=OFF", self.name),
            CxxOptionValue::Value(v) => format!("-D{}={}", self.name, v),
        }
    }

    pub fn cmake_value(&self) -> &'static str {
        match &self.value {
            CxxOptionValue::On => "ON",
            CxxOptionValue::Off => "OFF",
            CxxOptionValue::Value(v) => v,
        }
    }

    pub fn cxx(&self) -> String {
        match &self.value {
            CxxOptionValue::On => format!("-D{}=1", self.name),
            CxxOptionValue::Off => format!("-D{}=0", self.name),
            CxxOptionValue::Value(v) => format!("-D{}={}", self.name, v),
        }
    }

    pub fn enabled(&self) -> bool {
        match self.value {
            CxxOptionValue::On => true,
            CxxOptionValue::Off => false,
            CxxOptionValue::Value(_) => true,
        }
    }
}

mod cc_opts {
    use super::{CxxOption, CxxOptionValue};
    use std::sync::LazyLock;
    const ON: CxxOptionValue = super::CxxOptionValue::On;
    const OFF: CxxOptionValue = super::CxxOptionValue::Off;
    macro_rules! cmake_options {
        {$(option($name: ident $desc: tt $state: ident))*} => {
            $(
                #[doc = $desc]
                pub static $name: LazyLock<CxxOption> = LazyLock::<CxxOption>::new(|| { CxxOption {
                        name: stringify!($name),
                        value: option_env!(stringify!($name)).map(|item| item.into()).unwrap_or_else(|| $state),
                    }});
            )*
        }
    }

    cmake_options!(
        option(MNN_USE_SYSTEM_LIB "For opencl and vulkan, use system lib or use dlopen" OFF)
        option(MNN_BUILD_HARD "Build -mfloat-abi=hard or not" OFF)
        option(MNN_BUILD_SHARED_LIBS "MNN build shared or static lib" ON)
        option(MNN_WIN_RUNTIME_MT "MNN use /MT on Windows dll" OFF)
        option(MNN_FORBID_MULTI_THREAD "Disable Multi Thread" OFF)
        option(MNN_OPENMP "Use OpenMP's thread pool implementation. Does not work on iOS or Mac OS" OFF)
        option(MNN_USE_THREAD_POOL "Use MNN's own thread pool implementation" ON)
        option(MNN_BUILD_TRAIN "Build MNN's training framework" OFF)
        option(MNN_BUILD_DEMO "Build demo/exec or not" OFF)
        option(MNN_BUILD_TOOLS "Build tools/cpp or not" ON)
        option(MNN_BUILD_QUANTOOLS "Build Quantized Tools or not" OFF)
        option(MNN_EVALUATION "Build Evaluation Tools or not" OFF)
        option(MNN_BUILD_CONVERTER "Build Converter" OFF)
        option(MNN_SUPPORT_DEPRECATED_OP "Enable MNN's tflite quantized op" ON)
        option(MNN_DEBUG_MEMORY "MNN Debug Memory Access" OFF)
        option(MNN_DEBUG_TENSOR_SIZE "Enable Tensor Size" OFF)
        option(MNN_GPU_TRACE "Enable MNN Gpu Debug" OFF)
        option(MNN_SUPPORT_RENDER "Enable MNN Render Ops" OFF)
        option(MNN_SUPPORT_TRANSFORMER_FUSE "Enable MNN transformer Fuse Ops" OFF)
        option(MNN_PORTABLE_BUILD "Link the static version of third party libraries where possible to improve the portability of built executables" OFF)
        option(MNN_SEP_BUILD "Build MNN Backends and expression separately. Only works with MNN_BUILD_SHARED_LIBS=ON" ON)
        option(NATIVE_LIBRARY_OUTPUT "Native Library Path" OFF)
        option(NATIVE_INCLUDE_OUTPUT "Native Include Path" OFF)
        option(MNN_AAPL_FMWK "Build MNN.framework instead of traditional .a/.dylib" OFF)
        option(MNN_WITH_PLUGIN "Build with plugin op support." OFF)
        option(MNN_BUILD_MINI "Build MNN-MINI that just supports fixed shape models." OFF)
        option(MNN_USE_SSE "Use SSE optimization for x86 if possiable" ON)
        option(MNN_BUILD_CODEGEN "Build with codegen" OFF)
        option(MNN_ENABLE_COVERAGE "Build with coverage enable" OFF)
        option(MNN_BUILD_PROTOBUFFER "Build with protobuffer in MNN" ON)
        option(MNN_BUILD_OPENCV "Build OpenCV api in MNN." OFF)
        option(MNN_BUILD_LLM "Build llm library based MNN." OFF)
        option(MNN_BUILD_DIFFUSION "Build diffusion demo based MNN." OFF)
        option(MNN_INTERNAL "Build with MNN internal features, such as model authentication, metrics logging" OFF)
        option(MNN_JNI "Build MNN Jni for java to use" OFF)
        option(MNN_SUPPORT_BF16 "Enable MNN's bf16 op" OFF)
        option(MNN_LOW_MEMORY "Build MNN support low memory for weight quant model." OFF)
    );
}
