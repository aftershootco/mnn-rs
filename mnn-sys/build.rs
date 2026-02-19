use std::path::{Path, PathBuf};

use anyhow::*;
use build_target::Os;
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;

#[path = "build/bindgen.rs"]
mod bindgen;
#[path = "build/compile.rs"]
mod compile;
#[path = "build/download.rs"]
mod download;
#[path = "build/options.rs"]
mod options;

use bindgen::{mnn_c_bindgen, mnn_cpp_bindgen};
use compile::{build_cmake, mnn_c_build, prebuilt_lib_link};
use download::{download_mnn_source, download_prebuilt_mnn};
use options::{
    HALIDE_SEARCH, MANIFEST_DIR, MNN_COMPILE, TARGET_OS, TRACING_REPLACE, TRACING_SEARCH, VENDOR,
};

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
    let out_dir = PathBuf::from(std::env::var("OUT_DIR")?);
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=build/options.rs");
    println!("cargo:rerun-if-changed=build/download.rs");
    println!("cargo:rerun-if-changed=build/bindgen.rs");
    println!("cargo:rerun-if-changed=build/compile.rs");

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
                version,
                *options::TARGET_ARCH,
                *TARGET_OS
            )
        })?;
        let source = download_mnn_source(&version, &out_dir).with_context(|| {
            format!(
                "Failed to download MNN source for version {} for target {}-{}",
                version,
                *options::TARGET_ARCH,
                *TARGET_OS
            )
        })?;
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
    }
    println!("cargo:rustc-link-lib=static=MNN");
    Ok(())
}
