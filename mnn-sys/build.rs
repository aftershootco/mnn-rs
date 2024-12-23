use ::tap::*;
use error_stack::*;
#[derive(Debug, thiserror::Error)]
#[error("Failed to build mnn-sys")]
pub struct Error;
pub type Result<T, E = Report<Error>> = core::result::Result<T, E>;
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use std::{
    ffi::OsStr,
    path::{Path, PathBuf},
    sync::LazyLock,
};

static VENDOR: LazyLock<PathBuf> = LazyLock::new(|| {
    std::env::var("MNN_SRC")
        .map(PathBuf::from)
        .ok()
        .unwrap_or_else(|| PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/vendor")))
});
const MANIFEST_DIR: &str = env!("CARGO_MANIFEST_DIR");
static TARGET_POINTER_WIDTH: LazyLock<usize> = LazyLock::new(|| {
    std::env::var("CARGO_CFG_TARGET_POINTER_WIDTH")
        .expect("CARGO_CFG_TARGET_POINTER_WIDTH not set")
        .parse()
        .expect("Failed to parse CARGO_CFG_TARGET_POINTER_WIDTH")
});

static TARGET_FEATURES: LazyLock<Vec<String>> = LazyLock::new(|| {
    std::env::var("CARGO_CFG_TARGET_FEATURE")
        .expect("CARGO_CFG_TARGET_FEATURE not set")
        .split(',')
        .map(|s| s.to_string())
        .collect()
});

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

static MNN_COMPILE: LazyLock<bool> = LazyLock::new(|| {
    std::env::var("MNN_COMPILE")
        .ok()
        .and_then(|v| match v.as_str() {
            "1" | "true" | "yes" => Some(true),
            "0" | "false" | "no" => Some(false),
            _ => None,
        })
        .unwrap_or(true)
});

const HALIDE_SEARCH: &str =
    r#"HALIDE_ATTRIBUTE_ALIGN(1) halide_type_code_t code; // halide_type_code_t"#;
const TRACING_SEARCH: &str = "#define MNN_PRINT(format, ...) printf(format, ##__VA_ARGS__)\n#define MNN_ERROR(format, ...) printf(format, ##__VA_ARGS__)";
const TRACING_REPLACE: &str = r#"
enum class Level {
  Info = 0,
  Error = 1,
};
extern "C" {
void mnn_ffi_emit(const char *file, size_t line, Level level,
                  const char *message);
}
#define MNN_PRINT(format, ...)                                                 \
  {                                                                            \
    char logtmp[4096];                                                         \
    snprintf(logtmp, 4096, format, ##__VA_ARGS__);                             \
    mnn_ffi_emit(__FILE__, __LINE__, Level::Info, logtmp);                     \
  }

#define MNN_ERROR(format, ...)                                                 \
  {                                                                            \
    char logtmp[4096];                                                         \
    snprintf(logtmp, 4096, format, ##__VA_ARGS__);                             \
    mnn_ffi_emit(__FILE__, __LINE__, Level::Error, logtmp);                    \
  }
"#;

const STATIC_CRT: bool = cfg!(feature = "crt_static");

fn has_cpu_feature(feature: impl AsRef<str>) -> bool {
    let feature = feature.as_ref();
    TARGET_FEATURES.iter().any(|f| *f == feature)
}

fn ensure_vendor_exists(vendor: impl AsRef<Path>) -> Result<()> {
    if vendor
        .as_ref()
        .read_dir()
        .change_context(Error)
        .attach_printable_lazy(|| {
            format!("Vendor directory missing: {}", vendor.as_ref().display())
        })?
        .flatten()
        .count()
        == 0
    {
        return Err(Report::new(Error).attach_printable(
            "Vendor not found maybe you need to run \"git submodule update --init\"",
        ));
    }
    Ok(())
}

fn main() {
    match _main() {
        Ok(_) => (),
        Err(e) => {
            Report::set_color_mode(fmt::ColorMode::Color);
            Report::set_charset(fmt::Charset::default());
            eprintln!("{e:?}");
            panic!("Failed to compile mnn-sys")
        }
    }
}

fn _main() -> Result<()> {
    #[cfg(any(feature = "vulkan", feature = "coreml"))]
    compile_error!("Vulkan, CoreML are not supported currently");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=MNN_SRC");
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").change_context(Error)?);
    let source = (*VENDOR).clone();

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
        .change_context(Error)
        .attach_printable("Failed to copy vendor")?;
        let halide_runtime_h = vendor.join("include").join("MNN").join("HalideRuntime.h");
        #[cfg(unix)]
        std::fs::set_permissions(&halide_runtime_h, std::fs::Permissions::from_mode(0o644))
            .change_context(Error)?;

        let intptr_contents = std::fs::read_to_string(&halide_runtime_h).change_context(Error)?;
        let patched = intptr_contents.lines().collect::<Vec<_>>();
        if let Some(idx) = patched.iter().position(|line| line.contains(HALIDE_SEARCH)) {
            // remove the last line and the next 3 lines
            let patched = patched
                .into_iter()
                .enumerate()
                .filter(|(c_idx, _)| !(*c_idx == idx - 1 || (idx + 1..=idx + 3).contains(c_idx)))
                .map(|(_, c)| c)
                .collect::<Vec<_>>();

            std::fs::write(halide_runtime_h, patched.join("\n")).change_context(Error)?;
        }

        let mnn_define = vendor.join("include").join("MNN").join("MNNDefine.h");
        let patched = std::fs::read_to_string(&mnn_define)
            .change_context(Error)?
            .replace(TRACING_SEARCH, TRACING_REPLACE);
        #[cfg(unix)]
        std::fs::set_permissions(&mnn_define, std::fs::Permissions::from_mode(0o644))
            .change_context(Error)?;
        std::fs::write(mnn_define, patched).change_context(Error)?;
    }

    if *MNN_COMPILE {
        mnn_cpp_build(&vendor)?;
    } else if let core::result::Result::Ok(lib_dir) = std::env::var("MNN_LIB_DIR") {
        println!("cargo:rustc-link-search=native={}", lib_dir);
        println!("cargo:rustc-link-lib=static=MNN");
    } else {
        panic!("MNN_LIB_DIR not set while MNN_COMPILE is false");
    }

    mnn_c_build(PathBuf::from(MANIFEST_DIR).join("mnn_c"), &vendor)
        .attach_printable("Failed to build mnn_c")?;
    mnn_c_bindgen(&vendor, &out_dir).attach_printable("Failed to generate mnn_c bindings")?;
    mnn_cpp_bindgen(&vendor, &out_dir).attach_printable("Failed to generate mnn_cpp bindings")?;
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
    } else if *TARGET_OS == "linux" {
        #[cfg(feature = "opencl")]
        {
            if pkg_config::probe_library("OpenCL").is_err() {
                println!("cargo:rustc-link-lib=static=OpenCL");
            };
        }
    }
    if is_emscripten() {
        let emscripten_cache = std::process::Command::new("em-config")
            .arg("CACHE")
            .output()
            .change_context(Error)?
            .stdout;
        let emscripten_cache = std::str::from_utf8(&emscripten_cache)
            .change_context(Error)?
            .trim();
        let wasm32_emscripten_libs =
            PathBuf::from(emscripten_cache).join("sysroot/lib/wasm32-emscripten");
        println!(
            "cargo:rustc-link-search=native={}",
            wasm32_emscripten_libs.display()
        );
    }
    Ok(())
}

pub fn mnn_c_bindgen(vendor: impl AsRef<Path>, out: impl AsRef<Path>) -> Result<()> {
    let vendor = vendor.as_ref();
    let mnn_c = PathBuf::from(MANIFEST_DIR).join("mnn_c");
    mnn_c
        .read_dir()
        .change_context(Error)?
        .flatten()
        .for_each(|e| {
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
        .generate()
        .change_context(Error)?;
    bindings
        .write_to_file(out.as_ref().join("mnn_c.rs"))
        .change_context(Error)?;
    Ok(())
}

pub fn mnn_cpp_bindgen(vendor: impl AsRef<Path>, out: impl AsRef<Path>) -> Result<()> {
    let vendor = vendor.as_ref();
    let bindings = bindgen::Builder::default()
        .clang_args(["-x", "c++"])
        .clang_args(["-std=c++14"])
        .clang_arg(CxxOption::VULKAN.cxx())
        .clang_arg(CxxOption::METAL.cxx())
        .clang_arg(CxxOption::COREML.cxx())
        .clang_arg(CxxOption::OPENCL.cxx())
        .clang_arg(format!("-I{}", vendor.join("include").to_string_lossy()))
        .generate_cstr(true)
        .generate_inline_functions(true)
        .size_t_is_usize(true)
        .emit_diagnostics()
        .ctypes_prefix("core::ffi")
        .header(
            vendor
                .join("include")
                .join("MNN")
                .join("Interpreter.hpp")
                .to_string_lossy(),
        )
        .allowlist_item(".*SessionInfoCode.*");
    // let cmd = bindings.command_line_flags().join(" ");
    // println!("cargo:warn=bindgen: {}", cmd);
    let bindings = bindings.generate().change_context(Error)?;
    bindings
        .write_to_file(out.as_ref().join("mnn_cpp.rs"))
        .change_context(Error)?;
    Ok(())
}

pub fn mnn_c_build(path: impl AsRef<Path>, vendor: impl AsRef<Path>) -> Result<()> {
    let mnn_c = path.as_ref();
    let files = mnn_c
        .read_dir()
        .change_context(Error)?
        .flatten()
        .map(|e| e.path())
        .filter(|e| {
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
            config
        })
        .cpp(true)
        .static_flag(true)
        .static_crt(STATIC_CRT)
        .files(files)
        .std("c++14")
        .try_compile("mnn_c")
        .change_context(Error)
        .attach_printable("Failed to compile mnn_c library")?;
    Ok(())
}

pub fn rerun_if_changed(path: impl AsRef<Path>) {
    println!("cargo:rerun-if-changed={}", path.as_ref().display());
}

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

macro_rules! cxx_option_from_features {
    ($name:ident=> $feature: literal, $cxx:literal) => {
        pub const $name: CxxOption = cxx_option_from_feature!($feature, $cxx);
    };
    ($( $name:ident => $feature:literal, $cxx:literal),*) => {
        $(cxx_option_from_features!($name=> $feature, $cxx);)*

        pub fn all() -> Vec<CxxOption> {
            vec![$(Self::$name),*]
        }
    };
}
impl CxxOption {
    const fn from_bool(name: &'static str, value: bool) -> Self {
        Self {
            name,
            value: CxxOptionValue::from_bool(value),
        }
    }

    cxx_option_from_features! {
        VULKAN => "vulkan", "MNN_VULKAN",
        METAL => "metal", "MNN_METAL",
        COREML => "coreml", "MNN_COREML",
        OPENCL => "opencl", "MNN_OPENCL",
        CRT_STATIC => "crt_static", "MNN_WIN_RUNTIME_MT",
        SPARSE_COMPUTE => "sparse-compute", "MNN_USE_SPARSE_COMPUTE",
        THREADPOOL => "mnn-threadpool", "MNN_USE_THREAD_POOL",
        MINI_BUILD => "mini-build", "MNN_BUILD_MINI",
        ARM82 => "arm82", "MNN_ARM82",
        BF16 => "bf16", "MNN_SUPPORT_BF16",
        AVX512 => "avx512", "MNN_AVX512",
        SSE => "sse", "MNN_USE_SSE",
        LOW_MEMORY => "low-memory", "MNN_LOW_MEMORY",
        NEON => "neon", "MNN_USE_NEON",
        CPU_WEIGHT_DEQUANT_GEMM => "cpu-weight-dequant-gemm", "MNN_CPU_WEIGHT_DEQUANT_GEMM"
    }

    pub fn define(&self, build: &mut cc::Build) {
        if self.enabled() {
            build.define(self.name, self.cc());
        }
    }

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

    pub fn cc(&self) -> &str {
        match &self.value {
            CxxOptionValue::On => "1",
            CxxOptionValue::Off => "0",
            CxxOptionValue::Value(v) => v,
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

fn is_arm() -> bool {
    TARGET_ARCH.starts_with("armv7")
        || TARGET_ARCH.starts_with("aarch64")
        || TARGET_ARCH.starts_with("arm64")
}

fn is_x86() -> bool {
    TARGET_ARCH.starts_with("x86")
    // || TARGET_ARCH.starts_with("i686")
    // || TARGET_ARCH.starts_with("i386")
}

fn read_dir(input: impl AsRef<Path>) -> impl Iterator<Item = PathBuf> {
    ignore::WalkBuilder::new(input)
        .max_depth(Some(1))
        .build()
        .flatten()
        .map(|e| e.into_path())
}

pub fn mnn_cpp_build(vendor: impl AsRef<Path>) -> Result<()> {
    let mut build = cc::Build::new();
    let vendor = vendor.as_ref();
    let mut includes = vec![
        vendor.join("include/"),
        vendor.join("source/"),
        vendor.join("schema/current/"),
        vendor.join("3rd_party/"),
        vendor.join("3rd_party/flatbuffers/include"),
        vendor.join("3rd_party/half"),
        vendor.join("3rd_party/OpenCLHeaders/"),
    ];

    build
        .includes(&includes)
        .cpp(true)
        .static_crt(STATIC_CRT)
        .static_flag(true)
        .std("c++11");

    if cfg!(unix) {
        build
            .flag_if_supported("-Wno-format")
            .flag_if_supported("-Wno-ignored-qualifiers")
            .flag_if_supported("-Wno-sign-compare")
            .flag_if_supported("-Wno-unused-but-set-variable")
            .flag_if_supported("-Wno-unused-function")
            .flag_if_supported("-Wno-unused-lambda-capture")
            .flag_if_supported("-Wno-unused-local-typedef")
            .flag_if_supported("-Wno-unused-parameter")
            .flag_if_supported("-Wno-unused-private-field")
            .flag_if_supported("-Wno-unused-variable");
    }
    if cfg!(windows) {
        build
            .flag_if_supported("/wd4267")
            .flag_if_supported("/wd4018")
            .flag_if_supported("/wd4251")
            .flag_if_supported("/wd4996")
            .flag_if_supported("/wd4244")
            .flag_if_supported("/wd4146")
            .flag_if_supported("/wd4129")
            .flag_if_supported("/wd4305")
            .flag_if_supported("/wd4275")
            .flag_if_supported("/wd4101");
    }

    let like_msvc = build.get_compiler().is_like_msvc();
    if like_msvc {
        build.flag_if_supported("/source-charset:utf-8");
    }

    // CxxOption::VULKAN.define(&mut build);
    // CxxOption::COREML.define(&mut build);
    CxxOption::METAL.define(&mut build);
    CxxOption::OPENCL.define(&mut build);
    CxxOption::CRT_STATIC.define(&mut build);
    CxxOption::SPARSE_COMPUTE.define(&mut build);
    CxxOption::THREADPOOL.define(&mut build);
    CxxOption::MINI_BUILD.define(&mut build);
    (is_arm() && has_cpu_feature("neon")).then(|| CxxOption::NEON.define(&mut build));
    (is_x86() && has_cpu_feature("sse")).then(|| CxxOption::SSE.define(&mut build));
    CxxOption::LOW_MEMORY.define(&mut build);
    CxxOption::CPU_WEIGHT_DEQUANT_GEMM.define(&mut build);

    let core_files_dir = vendor.join("source").join("core");
    let core_files = ignore::Walk::new(&core_files_dir)
        .flatten()
        .filter(|e| e.path().extension() == Some(OsStr::new("cpp")))
        .map(|e| e.into_path());
    build.files(core_files);

    {
        let cpu_files_dir = vendor.join("source").join("backend").join("cpu");
        let cpu_files = ignore::WalkBuilder::new(&cpu_files_dir)
            .add(cpu_files_dir.join("compute"))
            .max_depth(Some(1))
            .add_custom_ignore_filename("CPUImageProcess.hpp")
            .add_custom_ignore_filename("CPUImageProcess.cpp")
            .build()
            .flatten()
            .filter(|e| e.path().extension() == Some(OsStr::new("cpp")))
            .map(|e| e.into_path());

        if CxxOption::ARM82.enabled() && is_arm() {
            build.define("ENABLE_ARMV82", None);
        }

        if is_arm() {
            build.include(cpu_files_dir.join("arm"));
            includes.push(cpu_files_dir.join("arm"));
            arm(&mut build, cpu_files_dir.join("arm"))?;
        }

        if has_cpu_feature("sse") && is_x86() && CxxOption::SSE.enabled() {
            x86_64(&mut build, &includes, cpu_files_dir.join("x86_x64"))?;
        }

        build.files(cpu_files);
    }

    {
        let cv_files_dir = vendor.join("source").join("cv");
        let cv_files = ignore::Walk::new(&cv_files_dir)
            .flatten()
            .filter(|e| e.path().extension() == Some(OsStr::new("cpp")))
            .map(|e| e.into_path());
        // build.include(cv_files_dir.join("schema").join("current"));
        if *TARGET_OS == "macos" {
            build.flag_if_supported("-fno-stack-check");
        }
        build.files(cv_files);
    }

    {
        let extra_files = ignore::WalkBuilder::new(vendor.join("source").join("math"))
            .add(vendor.join("source").join("shape"))
            .add(vendor.join("source").join("geometry"))
            .add(vendor.join("source").join("utils"))
            .build()
            .flatten()
            .filter(|p| cpp_filter(p.path()))
            .map(|e| e.into_path());
        build.files(extra_files);
    }

    #[cfg(feature = "opencl")]
    let build = opencl(build, vendor).change_context(Error)?;
    #[cfg(feature = "metal")]
    let build = metal(build, vendor).change_context(Error)?;

    build
        .try_compile("mnn")
        .change_context(Error)
        .attach_printable("Failed to compile mnn")?;
    Ok(())
}

fn arm(build: &mut cc::Build, arm_dir: impl AsRef<Path>) -> Result<&mut cc::Build> {
    let arm_source_dir = arm_dir.as_ref();

    let neon_sources: Vec<PathBuf> = vec![arm_source_dir.join("CommonOptFunctionNeon.cpp")];
    // if CxxOption::BF16.enabled() {
    //     let path = arm_source_dir.join("CommonNeonBF16.cpp");
    //     if path.exists() {
    //         neon_sources.push(path);
    //     }
    // }

    if *TARGET_POINTER_WIDTH == 64 {
        let arm64_sources_dir = arm_source_dir.join("arm64");
        let arm64_sources = ignore::Walk::new(&arm64_sources_dir)
            .flatten()
            .filter(|e| {
                e.path().extension() == Some(OsStr::new("S"))
                    || e.path().extension() == Some(OsStr::new("s"))
            })
            .map(|e| e.into_path());

        // MNN_LOW_MEMORY
        // MNN_CPU_WEIGHT_DEQUANT_GEMM

        build
            .files(arm64_sources.chain(neon_sources))
            .define("__aarch64__", None);
    } else if *TARGET_POINTER_WIDTH == 32 {
        let arm32_sources_dir = arm_source_dir.join("arm32");
        let arm32_sources = ignore::Walk::new(&arm32_sources_dir)
            .flatten()
            .filter(|e| {
                e.path().extension() == Some(OsStr::new("S"))
                    || e.path().extension() == Some(OsStr::new("s"))
            })
            .map(|e| e.into_path());

        // MNN_LOW_MEMORY
        // MNN_CPU_WEIGHT_DEQUANT_GEMM

        build
            .files(arm32_sources.chain(neon_sources))
            .define("__arm__", None);
    }
    Ok(build)
}

fn x86_64<'a>(
    build: &'a mut cc::Build,
    includes: &'_ [PathBuf],
    x86_64_dir: impl AsRef<Path>,
) -> Result<&'a mut cc::Build> {
    let mnn_assembler = std::env::var("MNN_ASSEMBLER").ok();
    let like_msvc = build.get_compiler().is_like_msvc();
    let win_use_asm = like_msvc && *TARGET_POINTER_WIDTH == 64 && mnn_assembler.is_some();
    let has_avx512 = target_has_avx512();
    let x86_src_dir = x86_64_dir.as_ref();
    let mnn_x8664_src = read_dir(&x86_src_dir).filter(|p| cpp_filter(p));
    let mnn_avx_src = read_dir(x86_src_dir.join("avx")).filter(|p| cpp_filter(p) || asm_filter(p));
    let mnn_avxfma_src =
        read_dir(x86_src_dir.join("avxfma")).filter(|p| cpp_filter(p) || asm_filter(p));
    let mnn_sse_src = read_dir(x86_src_dir.join("sse")).filter(|p| cpp_filter(p));
    let mnn_avx512_vnni_src = x86_src_dir.join("avx512/GemmInt8_VNNI.cpp");
    let mnn_avx512_src = read_dir(x86_src_dir.join("avx512"))
        .filter(|p| cpp_filter(p) || asm_filter(p))
        .filter(|p| p != &mnn_avx512_vnni_src);

    if has_avx512 && CxxOption::AVX512.enabled() && (!like_msvc || win_use_asm) {
        let mnn_avx512 = cc::Build::new()
            .files(mnn_avx512_src)
            .static_crt(STATIC_CRT)
            .static_flag(true)
            .define("MNN_USE_SSE", None)
            .define("MNN_X86_USE_ASM", None)
            .tap_mut(|build| {
                if build.get_compiler().is_like_msvc() {
                    build.flag_if_supported("/arch:AVX512");
                } else {
                    // target_compile_options(MNNAVX512 PRIVATE -m64 -mavx512f -mavx512dq -mavx512vl -mavx512bw -mfma)
                    build
                        .flag_if_supported("-m64")
                        .flag_if_supported("-mavx512f")
                        .flag_if_supported("-mavx512dq")
                        .flag_if_supported("-mavx512vl")
                        .flag_if_supported("-mavx512bw")
                        .flag_if_supported("-mfma");
                }
            })
            .try_compile_intermediates()
            .change_context(Error)?;
        build.objects(mnn_avx512);
        let mnn_avx512_vnni = true;
        if mnn_avx512_vnni {
            let mnn_avx512_vnni = cc::Build::new()
                .file(mnn_avx512_vnni_src)
                .define("MNN_USE_SSE", None)
                .define("MNN_X86_USE_ASM", None)
                .tap_mut(|build| {
                    if build.get_compiler().is_like_msvc() {
                        build.flag_if_supported("/arch:AVX512");
                    } else {
                        // target_compile_options(MNNAVX512_VNNI PRIVATE -m64 -mavx512f -mavx512dq -mavx512vl -mavx512bw -mfma -mavx512vnni)
                        build
                            .flag_if_supported("-m64")
                            .flag_if_supported("-mavx512f")
                            .flag_if_supported("-mavx512dq")
                            .flag_if_supported("-mavx512vl")
                            .flag_if_supported("-mavx512bw")
                            .flag_if_supported("-mfma")
                            .flag_if_supported("-mavx512vnni");
                    }
                })
                .try_compile_intermediates()
                .change_context(Error)?;
            build.objects(mnn_avx512_vnni);
        }
    }

    let mnn_sse = cc::Build::new()
        .cpp(true)
        .std("c++11")
        .includes(includes)
        .files(mnn_sse_src)
        .static_crt(STATIC_CRT)
        .static_flag(true)
        .define("MNN_USE_SSE", None)
        .tap_mut(|build| {
            if !like_msvc {
                build.flag_if_supported("-msse4.1");
            }
            CxxOption::LOW_MEMORY.define(build);
        })
        .try_compile_intermediates()
        .change_context(Error)
        .attach_printable("Failed to build sse extensions")?;

    let mnn_avx = cc::Build::new()
        .cpp(true)
        .std("c++11")
        .includes(includes)
        .files(mnn_avx_src)
        .static_crt(STATIC_CRT)
        .static_flag(true)
        .define("MNN_USE_SSE", None)
        .tap_mut(|build| {
            if like_msvc {
                build.flag_if_supported("/arch:AVX");
            } else {
                build
                    .flag_if_supported("-mavx2")
                    .define("MNN_X86_USE_ASM", None);
            }
            CxxOption::LOW_MEMORY.define(build);
        })
        .try_compile_intermediates()
        .change_context(Error)?;

    let mnn_avxfma = cc::Build::new()
        .cpp(true)
        .std("c++11")
        .includes(includes)
        .files(mnn_avxfma_src)
        .static_crt(STATIC_CRT)
        .static_flag(true)
        .define("MNN_USE_SSE", None)
        .tap_mut(|build| {
            if like_msvc {
                build.flag_if_supported("/arch:AVX2");
            } else {
                build
                    .flag_if_supported("-mavx2")
                    .flag_if_supported("-mfma")
                    .define("MNN_X86_USE_ASM", None);
            }
            CxxOption::LOW_MEMORY.define(build);
            CxxOption::BF16.define(build)
        })
        .try_compile_intermediates()
        .change_context(Error)?;

    let mnn_x8664 = cc::Build::new()
        .cpp(true)
        .std("c++11")
        .includes(includes)
        .files(mnn_x8664_src)
        .static_crt(STATIC_CRT)
        .static_flag(true)
        .tap_mut(|build| {
            CxxOption::LOW_MEMORY.define(build);
            CxxOption::SSE.define(build);
            CxxOption::CPU_WEIGHT_DEQUANT_GEMM.define(build);
            if has_avx512 && CxxOption::AVX512.enabled() && (!like_msvc || win_use_asm) {
                CxxOption::AVX512.define(build);
            }
        })
        .try_compile_intermediates()
        .change_context(Error)?;

    build.objects(mnn_sse);
    build.objects(mnn_x8664);
    build.objects(mnn_avx);
    build.objects(mnn_avxfma);

    has_avx512.then(|| {
        CxxOption::AVX512.define(build);
    });

    Ok(build)
}

fn target_has_avx512() -> bool {
    const AVX_PRG: &str = r#"
#ifndef __AVX512F__
#error "AVX-512 support is required to compile this program."
#endif
int main() {return 0;} "#;
    let out_dir: PathBuf = std::env::var("OUT_DIR")
        .expect("OUT_DIR must be set in build.rs")
        .into();
    std::fs::write(out_dir.join("test.c"), AVX_PRG).expect("Failed to write to out_dir");
    cc::Build::new()
        .file("test.c")
        .cargo_warnings(false)
        .try_compile("avx512")
        .is_ok()
}

fn cpp_filter(path: impl AsRef<Path>) -> bool {
    path.as_ref().extension() == Some(OsStr::new("cpp"))
        || path.as_ref().extension() == Some(OsStr::new("cc"))
}

fn asm_filter(path: impl AsRef<Path>) -> bool {
    path.as_ref().extension() == Some(OsStr::new("S"))
        || path.as_ref().extension() == Some(OsStr::new("s"))
}

pub trait HasExtension {
    fn has_extension<I: AsRef<OsStr>>(&self, extensions: impl IntoIterator<Item = I>) -> bool;
    fn has_extension_ignore_case<I: AsRef<OsStr>>(
        &self,
        extension: impl IntoIterator<Item = I>,
    ) -> bool;
}

impl<P: AsRef<Path>> HasExtension for P {
    fn has_extension<I: AsRef<OsStr>>(&self, extensions: impl IntoIterator<Item = I>) -> bool {
        let path_ext = self.as_ref().extension();
        extensions
            .into_iter()
            .any(|ext| path_ext == Some(ext.as_ref()))
    }
    fn has_extension_ignore_case<I: AsRef<OsStr>>(
        &self,
        extension: impl IntoIterator<Item = I>,
    ) -> bool {
        let path_ext = self.as_ref().extension().map(|ext| ext.as_encoded_bytes());
        extension.into_iter().any(|ext| {
            let ext = ext.as_ref().as_encoded_bytes();
            path_ext
                .map(|p| p.eq_ignore_ascii_case(ext))
                .unwrap_or_default()
        })
    }
}

pub fn metal(mut build: cc::Build, vendor: impl AsRef<Path>) -> Result<cc::Build> {
    let metal_source_dir = vendor.as_ref().join("source/backend/metal");
    let metal_files = ignore::Walk::new(&metal_source_dir)
        .flatten()
        .filter(|e| e.path().has_extension(["mm", "cpp"]))
        .map(ignore::DirEntry::into_path)
        .chain(core::iter::once(
            metal_source_dir.join("MetalOPRegister.mm"),
        ));

    cc_builder()
        .define(CxxOption::METAL.name, CxxOption::METAL.cc())
        .files(metal_files)
        .includes(mnn_includes(vendor))
        .flag("-fobjc-arc")
        .define("MNN_METAL_ENABLED", "1")
        .try_compile("MNNMetal")
        .change_context(Error)
        .attach_printable("Failed to compile MNNMetal")?;
    build.define("MNN_METAL_ENABLED", "1");
    Ok(build)
}

pub fn opencl(mut build: cc::Build, vendor: impl AsRef<Path>) -> Result<cc::Build> {
    let source = vendor.as_ref().join("source");
    let backend = source.join("backend");
    let opencl_files_dir = backend.join("opencl");
    let opencl_files = ignore::Walk::new(&opencl_files_dir)
        .flatten()
        .filter(|e| e.path().has_extension(["cpp"]))
        .map(|e| e.into_path());
    let ocl_includes = opencl_files_dir.join("schema").join("current");
    cc_builder()
        .define(CxxOption::OPENCL.name, CxxOption::OPENCL.cc())
        .includes(mnn_includes(vendor))
        .include(ocl_includes.clone())
        .define("MNN_OPENCL_ENABLED", "1")
        .files(opencl_files.chain([opencl_files_dir.join("execution/cl/opencl_program.cc")]))
        .try_compile("MNNOpenCL")
        .change_context(Error)
        .attach_printable("Failed to build MNNOpenCL")?;
    build.define("MNN_OPENCL_ENABLED", "1");
    Ok(build)
}

pub fn mnn_includes(vendor: impl AsRef<Path>) -> Vec<PathBuf> {
    let vendor = vendor.as_ref();
    vec![
        vendor.join("include/"),
        vendor.join("source/"),
        vendor.join("schema/current/"),
        vendor.join("3rd_party/"),
        vendor.join("3rd_party/flatbuffers/include"),
        vendor.join("3rd_party/half"),
        vendor.join("3rd_party/OpenCLHeaders/"),
    ]
}

pub fn cc_builder() -> cc::Build {
    cc::Build::new()
        .cpp(true)
        .static_crt(STATIC_CRT)
        .static_flag(true)
        .std("c++11")
        .to_owned()
}
