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
    let glue_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("glue");
    autocxx_bindings(&glue_path, &vendor).with_context(|| "Failed to generate autocxx bridge")?;
    let built = build_cmake(&vendor)?;
    println!("cargo:include={vendor}/include", vendor = vendor.display());
    println!("cargo:rustc-link-search=native={}", built.display());
    Ok(())
}

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

pub fn autocxx_bindings(path: impl AsRef<Path>, vendor: impl AsRef<Path>) -> Result<()> {
    let inc_path = vendor.as_ref().join("include");
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut inc_paths = vec![inc_path.clone(), manifest_dir];
    // pub struct EmptyBuilderContext;
    // impl autocxx_engine::BuilderContext for EmptyBuilderContext {
    //     fn setup() {}
    //     fn get_dependency_recorder() -> Option<Box<dyn autocxx_engine::RebuildDependencyRecorder>> {
    //         None
    //     }
    // }

    // CFG.exported_header_prefixes.push("mnn-glue");
    // dbg!(cxx_build::CFG);
    let _gate = cxx_build::bridge("src/shared.rs");
    let out_dir = PathBuf::from(std::env::var("OUT_DIR")?);
    inc_paths.push(out_dir.join("cxxbridge").join("include"));

    // let shared =
    //     autocxx_engine::Builder::<'_, EmptyBuilderContext>::new("src/shared.rs", &inc_paths)
    //         .extra_clang_args(&["-std=c++14"])
    //         .custom_gendir(out_dir.join("autocxx-shared"))
    //         .suppress_system_headers(true)
    //         .build_listing_files()
    //         .context("Failed to generate autocxx bindings")?;
    // let cpp_files = shared
    //     .2
    //     .iter()
    //     .filter(|f| f.to_string_lossy().contains("include"))
    //     .next()
    //     .ok_or_else(|| anyhow!("Failed to find include files"))?;
    // let include_folder = cpp_files
    //     .ancestors()
    //     .find(|p| p.ends_with("include"))
    //     .ok_or_else(|| anyhow!("Failed to find include folder"))?;

    // inc_paths.push(include_folder.to_path_buf());
    let mut builder = autocxx_build::Builder::new("src/ffi.rs", &inc_paths)
        .extra_clang_args(&["-std=c++14"])
        .build()
        .context("Failed to generate autocxx bindings")?;
    builder
        .std("c++14")
        .file("glue/TensorGlue.cpp")
        .compile("mnn-autocxx-bridge");
    println!("cargo:rerun-if-changed=src/ffi.rs");
    println!("cargo:rustc-link-lib=mnn-autocxx-bridge");

    Ok(())
}

pub fn try_patch_interpreter(patch: impl AsRef<Path>, vendor: impl AsRef<Path>) -> Result<()> {
    let patch = patch.as_ref();
    println!("cargo:rerun-if-changed={}", patch.display());
    let patch = std::fs::read_to_string(patch)?;
    let patch = diffy::Patch::from_str(&patch)?;
    let vendor = vendor.as_ref();
    let interpreter_path = vendor.join("include").join("MNN").join("Interpreter.hpp");
    let interpreter = std::fs::read_to_string(&interpreter_path)?;
    let patched_interpreter =
        diffy::apply(&interpreter, &patch).context("Failed to apply patches using diffy")?;
    std::fs::write(interpreter_path, patched_interpreter)?;
    Ok(())
}
