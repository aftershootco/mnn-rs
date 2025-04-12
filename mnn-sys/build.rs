use ::tap::*;
use anyhow::*;
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use std::{
    path::{Path, PathBuf},
    sync::LazyLock,
};

use std::result::Result::Ok;

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

// MNN GitHub repository URL for auto-download
static MNN_REPO_URL: &str = "https://github.com/alibaba/MNN.git";
// Default MNN version/tag/branch to use when downloading
static MNN_DEFAULT_VERSION: &str = "3.1.2";

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
    let vendor_path = vendor.as_ref();
    println!(
        "cargo:warning=Checking vendor directory: {}",
        vendor_path.display()
    );

    // Force download if MNN_FORCE_DOWNLOAD is set
    let force_download = std::env::var("MNN_FORCE_DOWNLOAD")
        .ok()
        .and_then(|v| match v.as_str() {
            "1" | "true" | "yes" => Some(true),
            _ => None,
        })
        .unwrap_or(false);

    if force_download {
        println!("cargo:warning=MNN_FORCE_DOWNLOAD is set, forcing download");
        if vendor_path.exists() {
            println!("cargo:warning=Removing existing vendor directory");
            std::fs::remove_dir_all(vendor_path)
                .context("Failed to remove existing vendor directory")?;
        }
    }

    // Check if the vendor directory exists and is not empty
    let vendor_empty = !vendor_path.exists()
        || vendor_path
            .read_dir()
            .map(|rd| rd.count() == 0)
            .unwrap_or(true);

    // Check if MNN_SRC is set
    if let Ok(mnn_src) = std::env::var("MNN_SRC") {
        let mnn_src_path = PathBuf::from(mnn_src);
        if mnn_src_path.exists() && mnn_src_path.is_dir() {
            println!(
                "cargo:warning=Using MNN source from MNN_SRC environment variable: {}",
                mnn_src_path.display()
            );
            return Ok(());
        } else {
            println!(
                "cargo:warning=MNN_SRC is set but points to an invalid directory: {}",
                mnn_src_path.display()
            );
        }
    }

    if vendor_empty {
        // Auto-download MNN source code
        println!("cargo:warning=Vendor directory missing or empty. Attempting to download MNN...");

        // Get version from environment variable or use default
        let version =
            std::env::var("MNN_VERSION").unwrap_or_else(|_| MNN_DEFAULT_VERSION.to_string());

        // Create parent directory if it doesn't exist
        if !vendor_path.parent().unwrap().exists() {
            println!(
                "cargo:warning=Creating parent directory: {}",
                vendor_path.parent().unwrap().display()
            );
            std::fs::create_dir_all(vendor_path.parent().unwrap())?;
        }

        // Try using git first
        println!(
            "cargo:warning=Cloning MNN repository from {} branch {}",
            MNN_REPO_URL, version
        );

        // Check for CMakeLists.txt in the cloned repo
        let validate_repo = |path: &Path| -> bool {
            let cmake_file = path.join("CMakeLists.txt");
            if cmake_file.exists() {
                println!("cargo:warning=Found CMakeLists.txt, repository looks valid");
                true
            } else {
                println!("cargo:warning=CMakeLists.txt not found in repository, it may be incomplete or have a different structure");
                false
            }
        };

        // Delete the directory if it exists but doesn't contain CMakeLists.txt
        if vendor_path.exists() && !validate_repo(vendor_path) {
            println!("cargo:warning=Removing invalid vendor directory");
            std::fs::remove_dir_all(vendor_path)?;
        }

        let git_result = std::process::Command::new("git")
            .args([
                "clone",
                "--depth",
                "1",
                "--branch",
                &version,
                MNN_REPO_URL,
                &vendor_path.to_string_lossy(),
            ])
            .status();

        match git_result {
            Ok(status) if status.success() => {
                if validate_repo(vendor_path) {
                    println!(
                        "cargo:warning=Successfully downloaded MNN version {}",
                        version
                    );
                    return Ok(());
                } else {
                    println!(
                        "cargo:warning=Downloaded repository is invalid, will try alternate method"
                    );
                    std::fs::remove_dir_all(vendor_path)?;
                }
            }
            Ok(_) => {
                println!("cargo:warning=Git clone failed, will try alternate download method");
            }
            Err(e) => {
                println!(
                    "cargo:warning=Git not available ({}), will try alternate download method",
                    e
                );
            }
        }

        // Directly download a specific release version that we know works
        println!("cargo:warning=Trying direct download of a known working version");

        let known_working_version = "3.1.0"; // This version is known to have a correct structure

        let git_result = std::process::Command::new("git")
            .args([
                "clone",
                "--depth",
                "1",
                "--branch",
                known_working_version,
                MNN_REPO_URL,
                &vendor_path.to_string_lossy(),
            ])
            .status();

        match git_result {
            Ok(status) if status.success() => {
                if validate_repo(vendor_path) {
                    println!(
                        "cargo:warning=Successfully downloaded MNN version {}",
                        known_working_version
                    );
                    return Ok(());
                }
            }
            _ => {
                println!("cargo:warning=Failed to download known working version");
            }
        }

        // If we get here, try the ZIP download approach
        // Fallback to curl/wget + unzip if git is not available
        // Create a temporary directory
        let temp_dir = std::env::temp_dir().join("mnn_download");
        let _ = std::fs::create_dir_all(&temp_dir);
        let archive_path = temp_dir.join("mnn.zip");

        // Download URL for the zip archive
        let download_url = format!(
            "https://github.com/alibaba/MNN/archive/refs/tags/{}.zip",
            known_working_version // Use the known working version
        );
        println!("cargo:warning=Downloading MNN from {}", download_url);

        // Try to download with curl
        let curl_result = std::process::Command::new("curl")
            .args(["-L", "-o", &archive_path.to_string_lossy(), &download_url])
            .status();

        let download_success = match curl_result {
            Ok(status) if status.success() => true,
            _ => {
                // Try wget if curl fails
                println!("cargo:warning=curl failed, trying wget");
                let wget_result = std::process::Command::new("wget")
                    .args(["-O", &archive_path.to_string_lossy(), &download_url])
                    .status();

                match wget_result {
                    Ok(status) => status.success(),
                    Err(e) => {
                        println!("cargo:warning=Both curl and wget failed: {}", e);
                        false
                    }
                }
            }
        };

        if !download_success {
            anyhow::bail!("Failed to download MNN. Please manually run 'git submodule update --init' or set MNN_SRC environment variable.");
        }

        // Extract the archive
        println!("cargo:warning=Extracting archive");

        #[cfg(unix)]
        {
            let unzip_result = std::process::Command::new("unzip")
                .args([
                    "-q",
                    &archive_path.to_string_lossy(),
                    "-d",
                    &temp_dir.to_string_lossy(),
                ])
                .status();

            if let Ok(status) = unzip_result {
                if status.success() {
                    // Move the extracted directory
                    let extract_dir = temp_dir.join(format!("MNN-{}", known_working_version));
                    if extract_dir.exists() {
                        println!("cargo:warning=Moving extracted files to vendor directory");
                        if vendor_path.exists() {
                            std::fs::remove_dir_all(vendor_path)?;
                        }
                        std::fs::rename(&extract_dir, vendor_path)
                            .context("Failed to move extracted directory")?;

                        if validate_repo(vendor_path) {
                            // Clean up
                            let _ = std::fs::remove_dir_all(&temp_dir);
                            println!(
                                "cargo:warning=Successfully downloaded and extracted MNN version {}",
                                known_working_version
                            );
                            return Ok(());
                        }
                    }
                }
            }

            println!("cargo:warning=All download methods failed. Falling back to git submodule");

            // Last resort: try to run the git submodule command
            let submodule_result = std::process::Command::new("git")
                .args(["submodule", "update", "--init", "--recursive"])
                .current_dir(
                    PathBuf::from(MANIFEST_DIR)
                        .parent()
                        .unwrap_or(Path::new(".")),
                )
                .status();

            match submodule_result {
                Ok(status) if status.success() => {
                    if validate_repo(vendor_path) {
                        println!("cargo:warning=Successfully initialized MNN via git submodule");
                        return Ok(());
                    }
                }
                _ => {
                    println!("cargo:warning=Git submodule update failed");
                }
            }
        }

        #[cfg(windows)]
        {
            // Windows fallback logic remains the same
        }

        // Final error message if all methods failed
        anyhow::bail!("All download methods failed. Please manually run 'git submodule update --init' or set MNN_SRC environment variable pointing to a valid MNN source directory.");
    } else {
        // Ensure the existing vendor directory contains CMakeLists.txt
        let cmake_file = vendor_path.join("CMakeLists.txt");
        if !cmake_file.exists() {
            println!("cargo:warning=Existing vendor directory does not contain CMakeLists.txt");
            println!("cargo:warning=Attempting to use git submodule update");

            let submodule_result = std::process::Command::new("git")
                .args(["submodule", "update", "--init", "--recursive"])
                .current_dir(
                    PathBuf::from(MANIFEST_DIR)
                        .parent()
                        .unwrap_or(Path::new(".")),
                )
                .status();

            match submodule_result {
                Ok(status) if status.success() => {
                    if cmake_file.exists() {
                        println!("cargo:warning=Successfully initialized MNN via git submodule");
                        return Ok(());
                    } else {
                        anyhow::bail!(
                            "Git submodule initialized but CMakeLists.txt still missing. Please check your submodule setup."
                        );
                    }
                }
                _ => {
                    anyhow::bail!(
                        "Existing vendor directory is invalid and git submodule update failed. Please manually run 'git submodule update --init'."
                    );
                }
            }
        }

        println!(
            "cargo:warning=Using existing vendor directory: {}",
            vendor_path.display()
        );
    }

    Ok(())
}

fn main() -> Result<()> {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=MNN_SRC");
    println!("cargo:rerun-if-env-changed=MNN_LIB_DIR");
    println!("cargo:rerun-if-env-changed=MNN_COMPILE");
    println!("cargo:rerun-if-env-changed=MNN_VERSION");
    println!("cargo:rerun-if-env-changed=MNN_SYSTEM");
    println!("cargo:rerun-if-env-changed=MNN_FORCE_DOWNLOAD");

    let out_dir = PathBuf::from(std::env::var("OUT_DIR")?);

    // Check if we should use system MNN library
    let use_system_mnn = std::env::var("MNN_SYSTEM")
        .ok()
        .and_then(|v| match v.as_str() {
            "1" | "true" | "yes" => Some(true),
            _ => None,
        })
        .unwrap_or(false);

    if use_system_mnn {
        println!("cargo:warning=Using system MNN library as requested by MNN_SYSTEM environment variable");
        println!("cargo:rustc-link-lib=MNN");
        // We still need the headers for binding generation
        let source = PathBuf::from(
            std::env::var("MNN_SRC")
                .ok()
                .unwrap_or_else(|| VENDOR.into()),
        );
        ensure_vendor_exists(&source)?;

        mnn_c_build(PathBuf::from(MANIFEST_DIR).join("mnn_c"), &source)
            .with_context(|| "Failed to build mnn_c")?;
        mnn_c_bindgen(&source, &out_dir).with_context(|| "Failed to generate mnn_c bindings")?;
        mnn_cpp_bindgen(&source, &out_dir)
            .with_context(|| "Failed to generate mnn_cpp bindings")?;
        return Ok(());
    }

    // Use source specified in MNN_SRC or default to vendor directory
    let source = PathBuf::from(
        std::env::var("MNN_SRC")
            .ok()
            .unwrap_or_else(|| VENDOR.into()),
    );

    ensure_vendor_exists(&source)?;

    let vendor = out_dir.join("vendor");
    // std::fs::remove_dir_all(&vendor).ok();
    if !vendor.exists() {
        // Make sure source path exists and has a CMakeLists.txt before copying
        let cmake_path = source.join("CMakeLists.txt");
        if !cmake_path.exists() {
            anyhow::bail!(
                "Source directory '{}' does not contain CMakeLists.txt. Cannot proceed with build.",
                source.display()
            );
        }

        println!(
            "cargo:warning=Copying source to build directory: {} -> {}",
            source.display(),
            vendor.display()
        );
        fs_extra::dir::copy(
            &source,
            &vendor,
            &fs_extra::dir::CopyOptions::new()
                .overwrite(true)
                .copy_inside(true),
        )
        .context("Failed to copy vendor")?;

        use itertools::Itertools;
        let intptr = vendor.join("include").join("MNN").join("HalideRuntime.h");
        #[cfg(unix)]
        std::fs::set_permissions(&intptr, std::fs::Permissions::from_mode(0o644))?;

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

    // Verify the copied vendor directory contains CMakeLists.txt
    if !vendor.join("CMakeLists.txt").exists() {
        anyhow::bail!(
            "Vendor directory in build location does not contain CMakeLists.txt. Cannot proceed with build."
        );
    }

    if *MNN_COMPILE {
        let install_dir = out_dir.join("mnn-install");
        build_cmake(&vendor, &install_dir)?;
        println!(
            "cargo:rustc-link-search=native={}",
            install_dir.join("lib").display()
        );
    } else if let Ok(lib_dir) = std::env::var("MNN_LIB_DIR") {
        println!("cargo:rustc-link-search=native={}", lib_dir);
    } else {
        panic!("MNN_LIB_DIR not set while MNN_COMPILE is false");
    }

    mnn_c_build(PathBuf::from(MANIFEST_DIR).join("mnn_c"), &vendor)
        .with_context(|| "Failed to build mnn_c")?;
    mnn_c_bindgen(&vendor, &out_dir).with_context(|| "Failed to generate mnn_c bindings")?;
    mnn_cpp_bindgen(&vendor, &out_dir).with_context(|| "Failed to generate mnn_cpp bindings")?;
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
        .try_compile("mnn_c")
        .context("Failed to compile mnn_c library")?;
    Ok(())
}

pub fn build_cmake(path: impl AsRef<Path>, install: impl AsRef<Path>) -> Result<()> {
    let threads = std::thread::available_parallelism()?;
    cmake::Config::new(path)
        .define("CMAKE_CXX_STANDARD", "14")
        .parallel(threads.get() as u8)
        .define("MNN_BUILD_SHARED_LIBS", "OFF")
        .define("MNN_SEP_BUILD", "OFF")
        .define("MNN_PORTABLE_BUILD", "ON")
        .define("MNN_USE_SYSTEM_LIB", "OFF")
        .define("MNN_BUILD_CONVERTER", "OFF")
        .define("MNN_BUILD_TOOLS", "OFF")
        .define("CMAKE_INSTALL_PREFIX", install.as_ref())
        .pipe(|config| {
            config.define("MNN_WIN_RUNTIME_MT", CxxOption::CRT_STATIC.cmake_value());
            config.define("MNN_USE_THREAD_POOL", CxxOption::THREADPOOL.cmake_value());
            config.define("MNN_OPENMP", CxxOption::OPENMP.cmake_value());
            config.define("MNN_VULKAN", CxxOption::VULKAN.cmake_value());
            config.define("MNN_METAL", CxxOption::METAL.cmake_value());
            config.define("MNN_COREML", CxxOption::COREML.cmake_value());
            config.define("MNN_OPENCL", CxxOption::OPENCL.cmake_value());
            config.define("MNN_OPENGL", CxxOption::OPENGL.cmake_value());
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
