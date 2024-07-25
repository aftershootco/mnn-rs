use ::tap::*;
use anyhow::*;
use std::path::{Path, PathBuf};
const VENDOR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/vendor");
const MANIFEST_DIR: &str = env!("CARGO_MANIFEST_DIR");

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
        try_patch_file(
            "patches/typedef_template.patch",
            &vendor.join("include").join("MNN").join("Interpreter.hpp"),
        )?;
        // try_patch_file(
        //     "patches/forward_guard.patch",
        //     &vendor.join("include/MNN/MNNForwardType.h"),
        // )?;
    }
    // let glue_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("glue");
    // autocxx_bindings(&glue_path, &vendor).with_context(|| "Failed to generate autocxx bridge")?;

    mnn_c_bindgen(&vendor, &out_dir).with_context(|| "Failed to generate mnn_c bindings")?;
    mnn_c_build(PathBuf::from(MANIFEST_DIR).join("mnn_c"), &vendor)
        .with_context(|| "Failed to build mnn_c")?;

    let built = build_cmake(&vendor)?;
    println!("cargo:include={vendor}/include", vendor = vendor.display());
    if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-lib=framework=Foundation");
        #[cfg(feature = "metal")]
        println!("cargo:rustc-link-lib=framework=CoreGraphics");
        #[cfg(feature = "metal")]
        println!("cargo:rustc-link-lib=framework=Metal");
        println!(
            "cargo:rustc-link-search=framework={}",
            built.join("build").display()
        );
        println!("cargo:rustc-link-lib=framework=MNN");
    } else {
        println!(
            "cargo:rustc-link-search=native={}",
            built.join("build").display()
        );
        println!("cargo:rustc-link-lib=static=MNN");
    }
    Ok(())
}

pub fn mnn_c_bindgen(vendor: impl AsRef<Path>, out: impl AsRef<Path>) -> Result<()> {
    let mnn_c = PathBuf::from(MANIFEST_DIR).join("mnn_c");
    mnn_c.read_dir()?.flatten().for_each(|e| {
        rerun_if_changed(e.path());
    });
    const HEADERS: &[&str] = &[
        "ErrorCode_c.h",
        "Interpreter_c.h",
        "Tensor_c.h",
        // "TensorUtils_c.h",
    ];

    let bindings = bindgen::Builder::default()
        // .pipe(|builder| {
        //     #[cfg(feature = "vulkan")]
        //     let builder = builder.clang_arg("-DMNN_VULKAN=1");
        //     #[cfg(feature = "metal")]
        //     let builder = builder.clang_arg("-DMNN_METAL=1");
        //     #[cfg(feature = "coreml")]
        //     let builder = builder.clang_arg("-DMNN_COREML=1");
        //     builder
        // })
        .detect_include_paths(true)
        .clang_arg(format!(
            "-I{}",
            vendor.as_ref().join("include").to_string_lossy()
        ))
        .pipe(|generator| {
            HEADERS.iter().fold(generator, |gen, header| {
                gen.header(mnn_c.join(header).to_string_lossy())
            })
        })
        .rustified_enum("MemoryMode")
        .rustified_enum("PowerMode")
        .rustified_enum("PrecisionMode")
        .rustified_enum("SessionMode")
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
        .generate()?;
    bindings.write_to_file(out.as_ref().join("mnn_c.rs"))?;
    Ok(())
}

pub fn mnn_c_build(path: impl AsRef<Path>, vendor: impl AsRef<Path>) -> Result<()> {
    let mnn_c = path.as_ref();
    cc::Build::new()
        .include(vendor.as_ref().join("include"))
        .cpp(true)
        .files(mnn_c.read_dir()?.flatten().map(|e| e.path()).filter(|e| {
            e.extension() == Some(std::ffi::OsStr::new("cpp"))
                || e.extension() == Some(std::ffi::OsStr::new("c"))
        }))
        .std("c++14")
        .try_compile("mnn_c")
        .context("Failed to compile mnn_c library")?;
    Ok(())
}
pub fn build_cmake(path: impl AsRef<Path>) -> Result<PathBuf> {
    let threads = std::thread::available_parallelism()?;
    Ok(cmake::Config::new(path)
        .parallel(threads.get() as u8)
        .cxxflag("-std=c++14")
        .define("MNN_BUILD_SHARED_LIBS", "OFF")
        .define("MNN_SEP_BUILD", "OFF")
        .define("MNN_PORTABLE_BUILD", "ON")
        .define("MNN_USE_SYSTEM_LIB", "OFF")
        .define("MNN_BUILD_CONVERTER", "OFF")
        .define("MNN_BUILD_TOOLS", "OFF")
        .define("MNN_AAPL_FMWK", "ON")
        .pipe(|_config| {
            #[cfg(feature = "vulkan")]
            _config.define("MNN_VULKAN", "ON");
            #[cfg(feature = "metal")]
            _config.define("MNN_METAL", "ON");
            #[cfg(feature = "coreml")]
            _config.define("MNN_COREML", "ON");
            _config
        })
        .build_target("MNN")
        .build())
}

// pub fn autocxx_bindings(path: impl AsRef<Path>, vendor: impl AsRef<Path>) -> Result<()> {
//     let inc_path = vendor.as_ref().join("include");
//     let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
//     let mut inc_paths = vec![inc_path.clone(), manifest_dir];
//     // pub struct EmptyBuilderContext;
//     // impl autocxx_engine::BuilderContext for EmptyBuilderContext {
//     //     fn setup() {}
//     //     fn get_dependency_recorder() -> Option<Box<dyn autocxx_engine::RebuildDependencyRecorder>> {
//     //         None
//     //     }
//     // }
//
//     // CFG.exported_header_prefixes.push("mnn-glue");
//     // dbg!(cxx_build::CFG);
//     let _gate = cxx_build::bridge("src/shared.rs");
//     let out_dir = PathBuf::from(std::env::var("OUT_DIR")?);
//     inc_paths.push(out_dir.join("cxxbridge").join("include"));
//
//     // let shared =
//     //     autocxx_engine::Builder::<'_, EmptyBuilderContext>::new("src/shared.rs", &inc_paths)
//     //         .extra_clang_args(&["-std=c++14"])
//     //         .custom_gendir(out_dir.join("autocxx-shared"))
//     //         .suppress_system_headers(true)
//     //         .build_listing_files()
//     //         .context("Failed to generate autocxx bindings")?;
//     // let cpp_files = shared
//     //     .2
//     //     .iter()
//     //     .filter(|f| f.to_string_lossy().contains("include"))
//     //     .next()
//     //     .ok_or_else(|| anyhow!("Failed to find include files"))?;
//     // let include_folder = cpp_files
//     //     .ancestors()
//     //     .find(|p| p.ends_with("include"))
//     //     .ok_or_else(|| anyhow!("Failed to find include folder"))?;
//
//     // inc_paths.push(include_folder.to_path_buf());
//     let mut builder = autocxx_build::Builder::new("src/ffi.rs", &inc_paths)
//         .extra_clang_args(&["-std=c++14"])
//         .build()
//         .context("Failed to generate autocxx bindings")?;
//     builder
//         .std("c++14")
//         .file("glue/TensorGlue.cpp")
//         .compile("mnn-autocxx-bridge");
//     println!("cargo:rerun-if-changed=src/ffi.rs");
//     println!("cargo:rustc-link-lib=mnn-autocxx-bridge");
//
//     Ok(())
// }
//
pub fn try_patch_file(patch: impl AsRef<Path>, file: impl AsRef<Path>) -> Result<()> {
    let patch = patch.as_ref();
    println!("cargo:rerun-if-changed={}", patch.display());
    let patch = std::fs::read_to_string(patch)?;
    let patch = diffy::Patch::from_str(&patch)?;
    // let vendor = vendor.as_ref();
    // let interpreter_path = vendor.join("include").join("MNN").join("Interpreter.hpp");
    let file_path = file.as_ref();
    let file = std::fs::read_to_string(&file_path)?;
    let patched_file =
        diffy::apply(&file, &patch).context("Failed to apply patches using diffy")?;
    std::fs::write(file_path, patched_file)?;
    Ok(())
}

pub fn rerun_if_changed(path: impl AsRef<Path>) {
    println!("cargo:rerun-if-changed={}", path.as_ref().display());
}
