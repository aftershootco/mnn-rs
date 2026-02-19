use anyhow::*;
use build_target::{Arch, Os};
use sha2::Digest as _;
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use std::{
    path::{Path, PathBuf},
    sync::LazyLock,
};
#[rustfmt::skip]
use ::tap::*;
const VENDOR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/vendor");
const MANIFEST_DIR: &str = env!("CARGO_MANIFEST_DIR");
static TARGET_ARCH: LazyLock<Arch> = LazyLock::new(|| build_target::target_arch());
static TARGET_OS: LazyLock<build_target::Os> = LazyLock::new(|| build_target::target_os());

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

const SUFFIXES: [&str; 5] = [
    "android_armv7_armv8_cpu_opencl_vulkan",
    "ios_armv82_cpu_metal_coreml",
    "linux_x64_cpu_opencl",
    "windows_x64_cpu_opencl",
    "macos_x64_arm82_cpu_opencl_metal",
];

const CHECKSUMS: [&str; 5] = [
    "sha256:f85050dfcab114da9d389c3a4dcde8421cdce5a767aab5dbd1a5f0debc8b704a",
    "sha256:2405ef73ab406844be9d16768a82dd76bec7aefaf05634eaad2f5d7202587aa0",
    "sha256:db42a3ed0eb4af791c872afc0fc82d9a13236a834c557c679fe4c9e39209129b",
    "sha256:2243dfea8e8364beed3fccb5be17b804d89feae91cbdd4ce577f147347f07555",
    "sha256:2bb04d451fe7587107d970322cbc80083c381bc50b06dd3ae3f2349eb5c82a89",
];

fn url_name_checksum(version: impl AsRef<str>) -> Result<(String, String, String)> {
    let version = version.as_ref();
    let pre_url =
        format!("https://github.com/alibaba/MNN/releases/download/{version}/mnn_{version}");

    let idx = match (&*TARGET_ARCH, &*TARGET_OS) {
        (&Arch::AArch64 | &Arch::Arm, &build_target::Os::Android) => 0,
        (&Arch::AArch64, &build_target::Os::iOS) => 1,
        (&Arch::X86_64, &build_target::Os::Linux) => 2,
        (&Arch::X86_64, &build_target::Os::Windows) => 3,
        (&Arch::X86_64 | &Arch::AArch64, &build_target::Os::MacOS) => 4,
        (arch, os) => anyhow::bail!("Prebuilt MNN is not available for target {}-{}", arch, os),
    };
    Ok((
        format!("{}_{}.zip", pre_url, SUFFIXES[idx]),
        format!("mnn_{version}_{}", SUFFIXES[idx]),
        CHECKSUMS[idx].to_string(),
    ))
}

fn verify_checksum(path: impl AsRef<Path>, expected: impl AsRef<str>) -> Result<()> {
    let expected = expected.as_ref();
    let mut file = std::fs::File::open(&path).with_context(|| {
        format!(
            "Failed to open file for checksum verification: {}",
            path.as_ref().display()
        )
    })?;
    let mut hasher = sha2::Sha256::new();
    std::io::copy(&mut file, &mut hasher).with_context(|| {
        format!(
            "Failed to read file for checksum verification: {}",
            path.as_ref().display()
        )
    })?;
    let actual = format!("sha256:{:x}", hasher.finalize());
    if actual != expected {
        anyhow::bail!(
            "Checksum mismatch for {}: expected {}, got {}",
            path.as_ref().display(),
            expected,
            actual
        );
    }
    Ok(())
}

fn download_prebuilt_mnn(version: impl AsRef<str>, out_dir: impl AsRef<Path>) -> Result<()> {
    let (url, root, checksum) = url_name_checksum(version)?;
    let dest = out_dir.as_ref().join("mnn_prebuilt");
    let dest_file = out_dir.as_ref().join("mnn_prebuilt.zip");
    if dest_file.exists() {
        // verify checksum
        eprintln!(
            "Prebuilt MNN zip already exists at {}, verifying checksum",
            dest_file.display()
        );
        verify_checksum(&dest_file, &checksum).with_context(|| {
            format!(
                "Checksum verification failed for existing prebuilt MNN at {}, expected checksum: {}",
                dest_file.display(),
                checksum
            )
        })?;
        eprintln!(
            "Prebuilt MNN already exists at {}, skipping download",
            dest_file.display()
        );
    } else {
        let response = reqwest::blocking::get(&url)
            .with_context(|| format!("Failed to download prebuilt MNN from {}", url))?;
        if !response.status().is_success() {
            anyhow::bail!(
                "Failed to download prebuilt MNN from {}, status: {}",
                url,
                response.status()
            );
        }
        let bytes = response
            .bytes()
            .with_context(|| format!("Failed to read response bytes from {}", url))?;
        std::fs::write(&dest_file, &bytes).with_context(|| {
            format!(
                "Failed to save prebuilt MNN zip from {} to {}",
                url,
                dest_file.display()
            )
        })?;
        verify_checksum(&dest_file, &checksum).with_context(|| {
            format!(
            "Checksum verification failed for downloaded prebuilt MNN at {}, expected checksum: {}",
            dest_file.display(),
            checksum
        )
        })?;
    }
    let file = std::fs::File::open(&dest_file).with_context(|| {
        format!(
            "Failed to open prebuilt MNN zip file at {} for extraction",
            dest_file.display()
        )
    })?;
    let mut zip = zip::ZipArchive::new(file)
        .with_context(|| format!("Failed to read zip archive from {}", url))?;
    zip.extract_unwrapped_root_dir(&dest, |path| path == Path::new(&root))
        .with_context(|| format!("Failed to extract MNN archive to {}", dest.display()))?;

    Ok(())
}

fn download_mnn_source(version: impl AsRef<str>, out_dir: impl AsRef<Path>) -> Result<()> {
    let version = version.as_ref();
    let url = format!(
        "https://api.github.com/repos/alibaba/MNN/zipball/{}",
        version
    );
    let dest = out_dir.as_ref().join("mnn_source");
    let dest_file = out_dir.as_ref().join("mnn_source.zip");
    if dest_file.exists() {
        eprintln!(
            "MNN source zip already exists at {}, skipping download",
            dest_file.display()
        );
        verify_checksum(&dest_file, "sha256:placeholder").with_context(|| {
            format!(
                "Checksum verification failed for existing MNN source at {}, expected checksum: sha256:placeholder",
                dest_file.display(),
            )
        })?;
        return Ok(());
    }
    let response = reqwest::blocking::get(&url)
        .with_context(|| format!("Failed to download MNN source from {}", url))?;
    if !response.status().is_success() {
        anyhow::bail!(
            "Failed to download MNN source from {}, status: {}",
            url,
            response.status()
        );
    }
    let bytes = response
        .bytes()
        .with_context(|| format!("Failed to read response bytes from {}", url))?;
    std::fs::write(&dest_file, &bytes).with_context(|| {
        format!(
            "Failed to save MNN source zip from {} to {}",
            url,
            dest_file.display()
        )
    })?;
    verify_checksum(&dest_file, "sha256:placeholder").with_context(|| {
            format!(
                "Checksum verification failed for downloaded MNN source at {}, expected checksum: sha256:placeholder",
                dest_file.display(),
            )
        })?;
    let file = std::fs::File::open(&dest_file).with_context(|| {
        format!(
            "Failed to open MNN source zip file at {} for extraction",
            dest_file.display()
        )
    })?;
    let mut zip = zip::ZipArchive::new(file)
        .with_context(|| format!("Failed to read zip archive from {}", url))?;
    zip.extract(&dest)
        .with_context(|| format!("Failed to extract MNN source archive to {}", dest.display()))?;
    Ok(())
}

fn prebuilt_lib_link(out_dir: impl AsRef<Path>) -> Result<()> {
    let prebuilt_dir = out_dir.as_ref().join("mnn_prebuilt");
    let is_debug = cfg!(debug_assertions);
    let debug_string = if is_debug { "Debug" } else { "Release" };
    match (&*TARGET_ARCH, &*TARGET_OS) {
        (&Arch::AArch64 | &Arch::Arm, &build_target::Os::Android) => {
            let arch = if *TARGET_ARCH == Arch::Arm {
                "armeabi-v7a"
            } else {
                "arm64-v8a"
            };
            println!(
                "cargo:rustc-link-search={}",
                prebuilt_dir.join(arch).display()
            );
            println!("cargo:rustc-link-lib=dylib=MNN");
            println!("cargo:rustc-link-lib=dylib=MNN_Vulkan");
            println!("cargo:rustc-link-lib=dylib=MNN_CL");
            println!("cargo:rustc-link-lib=dylib=c++_shared");
            println!("cargo:rustc-link-lib=dylib=mnncore");
        }
        (&Arch::AArch64, &build_target::Os::iOS) => {
            println!(
                "cargo:rustc-link-search={}",
                prebuilt_dir.join("Static").display()
            );
            println!("cargo:rustc-link-lib=dylib=MNN");
        }
        (&Arch::X86_64, &build_target::Os::Linux) => {
            println!(
                "cargo:rustc-link-search={}",
                prebuilt_dir.join("lib").join(debug_string).display()
            );
            println!("cargo:rustc-link-lib=static=MNN");
        }
        (&Arch::X86_64, &build_target::Os::Windows) => {
            let crt = if cfg!(feature = "crt_static") {
                "MT"
            } else {
                "MD"
            };
            println!(
                "cargo:rustc-link-search={}",
                prebuilt_dir
                    .join("lib")
                    .join("Release")
                    .join("static")
                    .join(crt)
                    .display()
            );
            println!("cargo:rustc-link-lib=static=MNN");
        }
        (&Arch::X86_64 | &Arch::AArch64, &build_target::Os::MacOS) => {
            println!(
                "cargo:rustc-link-search={}",
                prebuilt_dir.join("Static").display()
            );
            println!("cargo:rustc-link-lib=MNN");
        }
        (arch, os) => anyhow::bail!("Prebuilt MNN is not available for target {}-{}", arch, os),
    };
    Ok(())
}

fn main() -> Result<()> {
    let out_dir = PathBuf::from(std::env::var("OUT_DIR")?);
    println!("cargo:rerun-if-changed=build.rs");
    let source = PathBuf::from(
        std::env::var("MNN_SRC")
            .ok()
            .unwrap_or_else(|| VENDOR.into()),
    );

    if cfg!(feature = "download") {
        let version = std::env::var("MNN_VERSION").unwrap_or_else(|_| "3.4.0".to_string());
        download_prebuilt_mnn(&version, &out_dir).with_context(|| {
            format!(
                "Failed to download prebuilt MNN version {} for target {}-{}",
                version, *TARGET_ARCH, *TARGET_OS
            )
        })?;
        download_mnn_source(&version, &out_dir).with_context(|| {
            format!(
                "Failed to download MNN source for version {} for target {}-{}",
                version, *TARGET_ARCH, *TARGET_OS
            )
        })?;
        let source = out_dir.join("mnn_source");
        mnn_c_build(PathBuf::from(MANIFEST_DIR).join("mnn_c"), &source)
            .with_context(|| "Failed to build mnn_c from downloaded source")?;
        mnn_c_bindgen(&source, &out_dir)
            .with_context(|| "Failed to generate mnn_c bindings from downloaded source")?;
        mnn_cpp_bindgen(&source, &out_dir)
            .with_context(|| "Failed to generate mnn_cpp bindings from downloaded source")?;
        println!("cargo:include={source}/include", source = source.display());
        prebuilt_lib_link(&out_dir)?;
        return Ok(());
    }

    ensure_vendor_exists(&source)?;
    println!("cargo:rerun-if-env-changed=MNN_SRC");
    println!("cargo:rerun-if-env-changed=MNN_LIB_DIR");

    let vendor = out_dir.join("vendor");
    // std::fs::remove_dir_all(&vendor).ok();
    if !vendor.exists() {
        fs_extra::dir::copy(
            &source,
            &vendor,
            &fs_extra::dir::CopyOptions::new()
                .overwrite(true)
                .copy_inside(true),
        )
        .context("Failed to copy vendor")?;
        let intptr = vendor.join("include").join("MNN").join("HalideRuntime.h");
        #[cfg(unix)]
        std::fs::set_permissions(&intptr, std::fs::Permissions::from_mode(0o644))?;

        use itertools::Itertools;
        let intptr_contents = std::fs::read_to_string(&intptr)?;
        let patched = intptr_contents.lines().collect::<Vec<_>>();
        if let Some((idx, _)) = patched
            .iter()
            .find_position(|line| line.contains(HALIDE_SEARCH))
        {
            // remove the last line and the next 3 lines
            let patched = patched
                .into_iter()
                .enumerate()
                .filter(|(c_idx, _)| !(*c_idx == idx - 1 || (idx + 1..=idx + 3).contains(c_idx)))
                .map(|(_, c)| c)
                .collect::<Vec<_>>();

            std::fs::write(intptr, patched.join("\n"))?;
        }

        let mnn_define = vendor.join("include").join("MNN").join("MNNDefine.h");
        let patched =
            std::fs::read_to_string(&mnn_define)?.replace(TRACING_SEARCH, TRACING_REPLACE);
        #[cfg(unix)]
        std::fs::set_permissions(&mnn_define, std::fs::Permissions::from_mode(0o644))?;
        std::fs::write(mnn_define, patched)?;
    }

    if *MNN_COMPILE {
        let install_dir = out_dir.join("mnn-install");
        build_cmake(&vendor, &install_dir)?;
        println!(
            "cargo:rustc-link-search=native={}",
            install_dir.join("lib").display()
        );
    } else if let core::result::Result::Ok(lib_dir) = std::env::var("MNN_LIB_DIR") {
        println!("cargo:rustc-link-search=native={}", lib_dir);
    } else {
        panic!("MNN_LIB_DIR not set while MNN_COMPILE is false");
    }

    mnn_c_build(PathBuf::from(MANIFEST_DIR).join("mnn_c"), &vendor)
        .with_context(|| "Failed to build mnn_c")?;
    mnn_c_bindgen(&vendor, &out_dir).with_context(|| "Failed to generate mnn_c bindings")?;
    mnn_cpp_bindgen(&vendor, &out_dir).with_context(|| "Failed to generate mnn_cpp bindings")?;
    println!("cargo:include={vendor}/include", vendor = vendor.display());
    if *TARGET_OS == Os::MacOS {
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
    let bindings = bindings.generate()?;
    bindings.write_to_file(out.as_ref().join("mnn_cpp.rs"))?;
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
            config
        })
        .cpp(true)
        .files(files)
        .std("c++14")
        .try_compile("mnn_c")
        .context("Failed to compile mnn_c library")?;
    Ok(())
}

pub fn build_cmake(path: impl AsRef<Path>, install: impl AsRef<Path>) -> Result<()> {
    cmake::Config::new(path)
        .define("CMAKE_CXX_STANDARD", "14")
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
            #[cfg(all(target_os = "windows", target_env = "msvc"))]
            {
                config.profile("Release");
                config.define("CMAKE_MSVC_RUNTIME_LIBRARY", "MultiThreadedDLL");
            }
            config.define("MNN_WIN_RUNTIME_MT", CxxOption::CRT_STATIC.cmake_value());
            config.define("MNN_USE_THREAD_POOL", CxxOption::THREADPOOL.cmake_value());
            config.define("MNN_OPENMP", CxxOption::OPENMP.cmake_value());
            config.define("MNN_VULKAN", CxxOption::VULKAN.cmake_value());
            config.define("MNN_METAL", CxxOption::METAL.cmake_value());
            config.define("MNN_COREML", CxxOption::COREML.cmake_value());
            config.define("MNN_OPENCL", CxxOption::OPENCL.cmake_value());
            config.define("MNN_OPENGL", CxxOption::OPENGL.cmake_value());
            // config.define("CMAKE_CXX_FLAGS", "-O0");
            // #[cfg(windows)]
            if *TARGET_OS == Os::Windows {
                config.define("CMAKE_CXX_FLAGS", "-DWIN32=1");
            }
            config
        })
        .build();
    Ok(())
}

// pub fn try_patch_file(patch: impl AsRef<Path>, file: impl AsRef<Path>) -> Result<()> {
//     let patch = dunce::canonicalize(patch)?;
//     rerun_if_changed(&patch);
//     let patch = std::fs::read_to_string(&patch)?;
//     let patch = diffy::Patch::from_str(&patch)?;
//     let file_path = file.as_ref();
//     let file = std::fs::read_to_string(file_path).context("Failed to read input file")?;
//     let patched_file =
//         diffy::apply(&file, &patch).context("Failed to apply patches using diffy")?;
//     std::fs::write(file_path, patched_file)?;
//     Ok(())
// }

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
