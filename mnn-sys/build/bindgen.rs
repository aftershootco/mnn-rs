use std::path::{Path, PathBuf};

use anyhow::*;

use crate::options::{CxxOption, MANIFEST_DIR};

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

    let mut builder = bindgen::Builder::default()
        .clang_arg(CxxOption::VULKAN.cxx())
        .clang_arg(CxxOption::METAL.cxx())
        .clang_arg(CxxOption::COREML.cxx())
        .clang_arg(CxxOption::OPENCL.cxx())
        .clang_arg(format!("-I{}", vendor.join("include").to_string_lossy()));

    for header in HEADERS {
        builder = builder.header(mnn_c.join(header).to_string_lossy());
    }

    let bindings = builder
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

pub fn rerun_if_changed(path: impl AsRef<Path>) {
    println!("cargo:rerun-if-changed={}", path.as_ref().display());
}
