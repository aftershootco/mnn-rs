use std::path::Path;

use anyhow::*;
use build_target::Os;

use crate::options::{CxxOption, TARGET_OS};

/// Walk nvcc's symlink chain and return the first `<root>` (of `<root>/bin/nvcc`)
/// that holds the toolkit headers, i.e. a complete CUDA_TOOLKIT_ROOT_DIR.
#[cfg(feature = "cuda")]
fn locate_cuda_toolkit_root() -> Option<std::path::PathBuf> {
    let mut nvcc = std::env::split_paths(&std::env::var_os("PATH")?)
        .map(|dir| dir.join("nvcc"))
        .find(|p| p.is_file())?;
    loop {
        let bin = nvcc.parent()?;
        let root = bin.parent()?;
        if root.join("include/cuda_runtime.h").is_file() {
            return Some(root.to_owned());
        }
        let target = std::fs::read_link(&nvcc).ok()?;
        nvcc = if target.is_absolute() {
            target
        } else {
            bin.join(target)
        };
    }
}

/// Find the host compiler nvcc's own profile pins (`compiler-bindir` in
/// `nvcc.profile`, used by Nix to hand nvcc a supported gcc).
#[cfg(feature = "cuda")]
fn nvcc_pinned_host_compiler() -> Option<std::path::PathBuf> {
    let nvcc = std::env::split_paths(&std::env::var_os("PATH")?)
        .map(|dir| dir.join("nvcc"))
        .find(|p| p.is_file())?;
    let profile = std::fs::canonicalize(&nvcc)
        .ok()?
        .parent()?
        .join("nvcc.profile");
    let contents = std::fs::read_to_string(profile).ok()?;
    contents.lines().find_map(|line| {
        let (key, value) = line.split_once('=')?;
        if key.trim() != "compiler-bindir" {
            return None;
        }
        let gxx = std::path::Path::new(value.trim()).join("g++");
        gxx.is_file().then_some(gxx)
    })
}

pub fn mnn_c_build(path: impl AsRef<Path>, vendor: impl AsRef<Path>) -> Result<()> {
    let mnn_c = path.as_ref();
    let files = mnn_c.read_dir()?.flatten().map(|e| e.path()).filter(|e| {
        e.extension() == Some(std::ffi::OsStr::new("cpp"))
            || e.extension() == Some(std::ffi::OsStr::new("c"))
    });
    let vendor = vendor.as_ref();
    let mut build = cc::Build::new();
    build
        .include(vendor.join("include"))
        .cpp(true)
        .files(files)
        .std("c++14");

    #[cfg(feature = "vulkan")]
    build.define("MNN_VULKAN", "1");
    #[cfg(feature = "opengl")]
    build.define("MNN_OPENGL", "1");
    #[cfg(feature = "metal")]
    build.define("MNN_METAL", "1");
    #[cfg(feature = "coreml")]
    build.define("MNN_COREML", "1");
    #[cfg(feature = "opencl")]
    build.define("MNN_OPENCL", "ON");

    build
        .try_compile("mnn_c")
        .context("Failed to compile mnn_c library")?;
    Ok(())
}

pub fn build_cmake(path: impl AsRef<Path>, install: impl AsRef<Path>) -> Result<()> {
    let mut config = cmake::Config::new(path);
    // C++17 with cuda: MNN pins -std=c++11 into CMAKE_CXX_FLAGS unless the
    // standard is exactly 17, FindCUDA forwards that pin to nvcc, and current
    // libstdc++ headers (gcc >= 14) no longer compile as C++11 under nvcc's
    // EDG frontend.
    let cxx_standard = if cfg!(feature = "cuda") { "17" } else { "14" };
    config
        .define("CMAKE_CXX_STANDARD", cxx_standard)
        .define("MNN_BUILD_SHARED_LIBS", "OFF")
        .define("MNN_SEP_BUILD", "OFF")
        .define("MNN_PORTABLE_BUILD", "ON")
        .define("MNN_USE_SYSTEM_LIB", "OFF")
        .define("MNN_BUILD_CONVERTER", "OFF")
        .define("MNN_BUILD_TOOLS", "OFF")
        .define("CMAKE_INSTALL_PREFIX", install.as_ref())
        .define("MNN_WIN_RUNTIME_MT", CxxOption::CRT_STATIC.cmake_value())
        .define("MNN_USE_THREAD_POOL", CxxOption::THREADPOOL.cmake_value())
        .define("MNN_OPENMP", CxxOption::OPENMP.cmake_value())
        .define("MNN_VULKAN", CxxOption::VULKAN.cmake_value())
        .define("MNN_CUDA", CxxOption::CUDA.cmake_value())
        .define("MNN_METAL", CxxOption::METAL.cmake_value())
        .define("MNN_COREML", CxxOption::COREML.cmake_value())
        .define("MNN_OPENCL", CxxOption::OPENCL.cmake_value())
        .define("MNN_OPENGL", CxxOption::OPENGL.cmake_value());

    // MNN's legacy FindCUDA derives include/lib dirs from the toolkit root. On
    // split-package setups (e.g. Nix) nvcc lives apart from the cudart headers,
    // so let the caller point cmake at a complete toolkit.
    if let Some(root) = std::env::var_os("CUDA_TOOLKIT_ROOT_DIR") {
        config.define("CUDA_TOOLKIT_ROOT_DIR", root);
    }
    // Otherwise locate one: FindCUDA roots itself at nvcc's grandparent dir,
    // but nvcc is often reached through symlink farms (e.g. Nix system
    // profiles) whose roots lack the toolkit headers. Follow the chain and
    // pick the first root that actually contains cuda_runtime.h.
    #[cfg(feature = "cuda")]
    if std::env::var_os("CUDA_TOOLKIT_ROOT_DIR").is_none() {
        if let Some(root) = locate_cuda_toolkit_root() {
            config.define("CUDA_TOOLKIT_ROOT_DIR", root);
        }
    }
    // Escape hatch for extra nvcc flags, e.g. -allow-unsupported-compiler when
    // the host gcc is newer than the CUDA toolkit officially supports.
    if let Some(flags) = std::env::var_os("CUDA_NVCC_FLAGS") {
        config.define("CUDA_NVCC_FLAGS", flags);
    }
    // nvcc parses the host compiler's standard headers; point it at a gcc the
    // toolkit supports when the default host compiler is too new.
    if let Some(host) = std::env::var_os("CUDA_HOST_COMPILER") {
        config.define("CUDA_HOST_COMPILER", host);
    }
    // Nix pins a supported gcc in nvcc.profile (compiler-bindir), but FindCUDA
    // overrides it by passing -ccbin CMAKE_C_COMPILER unless CUDA_HOST_COMPILER
    // is set; mirror the pin so nvcc doesn't get handed a too-new gcc.
    #[cfg(feature = "cuda")]
    if std::env::var_os("CUDA_HOST_COMPILER").is_none() {
        if let Some(gxx) = nvcc_pinned_host_compiler() {
            config.define("CUDA_HOST_COMPILER", gxx);
        }
    }

    #[cfg(all(target_os = "windows", target_env = "msvc"))]
    {
        config.profile("Release");
        config.define("CMAKE_MSVC_RUNTIME_LIBRARY", "MultiThreadedDLL");
    }

    if *TARGET_OS == Os::Windows {
        config.define("CMAKE_CXX_FLAGS", "-DWIN32=1");
    }

    config.build();
    Ok(())
}
