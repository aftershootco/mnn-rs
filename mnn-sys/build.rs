use ::tap::*;
use anyhow::*;
use std::path::{Path, PathBuf};
const VENDOR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/vendor");

fn main() -> Result<()> {
    let out_dir = PathBuf::from(std::env::var("OUT_DIR")?);
    let vendor = out_dir.join("vendor");
    if !vendor.exists() {
        fs_extra::dir::copy(
            VENDOR,
            &vendor,
            &fs_extra::dir::CopyOptions::new()
                .overwrite(true)
                .copy_inside(true),
        )
        .context("Failed to copy vendor")?;
        try_patch_interpreter("patches/typedef_template.patch", &vendor)?;
    }
    autocxx_bindings(&vendor).with_context(|| "Failed to generate autocxx bridge")?;
    // build_interpreter_bridge(&vendor)?;
    // interpreter_bridge_bindings(&vendor)?;
    let built = build_cmake(&vendor)?;
    println!("cargo:include={vendor}/include", vendor = vendor.display());
    println!("cargo:rustc-link-search=native={}", built.display());
    Ok(())
}

// pub fn patch_vendor(repo: impl AsRef<Path>, patch: impl AsRef<Path>) -> Result<()> {
//     let patch_path = patch.as_ref();
//     let patch = git2::Diff::from_buffer(
//         &std::fs::read(&patch_path)
//             .with_context(|| format!("Failed to open patch file {patch_path:?}"))?,
//     )
//     .with_context(|| "Failed to parse patch file")?;
//     let repo_path = repo.as_ref();
//     let repo = git2::Repository::open(repo_path)
//         .with_context(|| format!("Failed to open repo {repo_path:?}"))?;
//     repo.apply(&patch, git2::ApplyLocation::WorkDir, None)?;
//     Ok(())
// }

pub fn build_cmake(path: impl AsRef<Path>) -> Result<PathBuf> {
    let threads = std::thread::available_parallelism()?;
    Ok(cmake::Config::new(path)
        .parallel(threads.get() as u8)
        .cxxflag("-std=c++14")
        .define("MNN_BUILD_SHARED_LIBS", "OFF")
        .define("MNN_SEP_BUILD", "OFF")
        .define("MNN_USE_SYSTEM_LIB", "OFF")
        .define("MNN_BUILD_CONVERTER", "OFF")
        .define("MNN_BUILD_TOOLS", "OFF")
        .pipe(|_config| {
            #[cfg(feature = "vulkan")]
            _config.define("MNN_VULKAN", "ON");
            #[cfg(feature = "metal")]
            _config.define("MNN_METAL", "ON");
            #[cfg(feature = "coreml")]
            _config.define("MNN_COREML", "ON");
            _config
        })
        .build())
}

// pub fn build_interpreter_bridge(path: impl AsRef<Path>) -> Result<()> {
//     cc::Build::new()
//         .cpp(true)
//         .file("bridge/interpreter.cpp")
//         .file("bridge/interpreter.h")
//         .include(path.as_ref().join("include"))
//         .emit_rerun_if_env_changed(true)
//         .compile("interpreter_bridge");
//     Ok(())
// }
// pub fn interpreter_bridge_bindings(path: impl AsRef<Path>) -> Result<()> {
//     let out_dir = PathBuf::from(std::env::var("OUT_DIR")?);
//
//     bindgen::builder()
//         .clang_args(&["-x", "c++", "-std=c++14"])
//         .clang_args(&[
//             "-I",
//             path.as_ref()
//                 .join("include")
//                 .to_str()
//                 .ok_or_else(|| anyhow!("Failed to convert path to string"))?,
//         ])
//         .header("bridge/interpreter.h")
//         .enable_cxx_namespaces()
//         .detect_include_paths(true)
//         .generate()?
//         .write_to_file(out_dir.join("bindings.rs"))?;
//     Ok(())
// }

pub fn autocxx_bindings(path: impl AsRef<Path>) -> Result<()> {
    let inc_path = path.as_ref().join("include");
    dbg!(&inc_path);
    let mut builder = autocxx_build::Builder::new("src/ffi.rs", &[&inc_path]).build()?;
    builder.std("c++14").compile("mnn-autocxx-bridge");
    println!("cargo:rerun-if-changed=src/ffi.rs");
    println!("cargo:rustc-link-lib=mnn-autocxx-bridge");

    Ok(())
}

pub fn try_patch_interpreter(patch: impl AsRef<Path>, vendor: impl AsRef<Path>) -> Result<()> {
    let patch = patch.as_ref();
    println!("cargo:rerun-if-changed={}", patch.display());
    let patch = std::fs::read_to_string(patch)?;
    let patch = diffy::Patch::from_str(&patch)?;

    // let patch = patchkit::patch::UnifiedPatch::parse_patch(patch.lines().map(|i| i.as_bytes()))
    //     .context("Failed to parse patch")?;
    let vendor = vendor.as_ref();
    let interpreter_path = vendor.join("include").join("MNN").join("Interpreter.hpp");
    let interpreter = std::fs::read_to_string(&interpreter_path)?;
    let patched_interpreter =
        diffy::apply(&interpreter, &patch).context("Failed to apply patches using diffy")?;

    std::fs::write(interpreter_path, patched_interpreter)?;
    Ok(())
}

// pub fn copy_and_patch_vendor(vendor: impl AsRef<Path>) -> Result<PathBuf> {
//     let vendor = vendor.as_ref();
//     let out_dir = PathBuf::from(std::env::var("OUT_DIR")?);
//     let vendor_out = out_dir.join("vendor");
//     std::fs::create_dir_all(&vendor_out)?;
//     fs_extra::dir::copy(
//         vendor,
//         &vendor_out,
//         &fs_extra::dir::CopyOptions::new()
//             .overwrite(true)
//             .copy_inside(true),
//     )
//     .context("Failed to copy vendor")?;
//     // std::fs::remove_file(vendor_out.join(".git")).ok();
//     let vendor_git_bare = concat!(
//         env!("CARGO_MANIFEST_DIR"),
//         "/../.git/modules/mnn-sys/vendor"
//     );
//     // let vendor_git_bare = std::fs::canonicalize(vendor_git_bare)?;
//     std::fs::write(
//         vendor_out.join(".git"),
//         format!("gitdir: {vendor_git_bare}"),
//     )?;
//     // fs_extra::dir::copy(
//     //     vendor_git_bare,
//     //     &vendor_out.join(".git"),
//     //     &fs_extra::dir::CopyOptions::new()
//     //         .overwrite(false)
//     //         .copy_inside(true),
//     // )
//     // .context("Failed to copy bare repo")
//     // .with_context(|| format!("From {vendor_git_bare}"))
//     // .with_context(|| format!("To {}", vendor_out.join(".git").display()))?;
//     patch_vendor(&vendor_out, "patches/typedef_template.patch")
//         .with_context(|| "Failed to patch vendor")?;
//     Ok(vendor_out)
// }
