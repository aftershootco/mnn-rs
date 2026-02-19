use std::path::Path;

use anyhow::*;
use build_target::Os;

use crate::options::{CxxOption, TARGET_OS};

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
    config
        .define("CMAKE_CXX_STANDARD", "14")
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
        .define("MNN_METAL", CxxOption::METAL.cmake_value())
        .define("MNN_COREML", CxxOption::COREML.cmake_value())
        .define("MNN_OPENCL", CxxOption::OPENCL.cmake_value())
        .define("MNN_OPENGL", CxxOption::OPENGL.cmake_value());

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

pub fn prebuilt_lib_link(out_dir: impl AsRef<Path>) -> Result<()> {
    use build_target::Arch;

    let prebuilt_dir = out_dir.as_ref().join("mnn_prebuilt");
    let is_debug = cfg!(debug_assertions);
    let debug_string = if is_debug { "Debug" } else { "Release" };

    match (&*crate::options::TARGET_ARCH, &*TARGET_OS) {
        (Arch::AArch64 | Arch::Arm, build_target::Os::Android) => {
            let arch = if *crate::options::TARGET_ARCH == Arch::Arm {
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
        (Arch::AArch64, build_target::Os::iOS) => {
            println!(
                "cargo:rustc-link-search={}",
                prebuilt_dir.join("Static").display()
            );
            println!("cargo:rustc-link-lib=dylib=MNN");
        }
        (Arch::X86_64, build_target::Os::Linux) => {
            println!(
                "cargo:rustc-link-search={}",
                prebuilt_dir.join("lib").join(debug_string).display()
            );
            println!("cargo:rustc-link-lib=static=MNN");
        }
        (Arch::X86_64, build_target::Os::Windows) => {
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
        (Arch::X86_64 | Arch::AArch64, build_target::Os::MacOS) => {
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
