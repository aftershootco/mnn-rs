pub fn main() {
    // #[cfg(target_os = "emscripten")]
    {
        println!("cargo:rustc-link-arg=-fvisibility=default");
        println!("cargo:rustc-link-arg=-sALLOW_MEMORY_GROWTH=1");
        println!("cargo:rustc-link-arg=-sERROR_ON_UNDEFINED_SYMBOLS=0");
        println!("cargo:rustc-link-arg=-sDEFAULT_LIBRARY_FUNCS_TO_INCLUDE=$readAsmConstArgs");
    }
}
