// use std::sync::LazyLock;
// static TARGET_OS: LazyLock<String> =
//     LazyLock::new(|| std::env::var("CARGO_CFG_TARGET_OS").expect("CARGO_CFG_TARGET_OS not set"));
// pub fn main() {
//     // #[cfg(target_os = "emscripten")]
//     if *TARGET_OS == "emscripten" {
//         println!("cargo:rustc-link-arg=-fvisibility=default");
//         println!("cargo:rustc-link-arg=-sALLOW_MEMORY_GROWTH=1");
//         // println!("cargo:rustc-link-arg=-sERROR_ON_UNDEFINED_SYMBOLS=0");
//         // println!("cargo:rustc-link-arg=-sDEFAULT_LIBRARY_FUNCS_TO_INCLUDE=$readAsmConstArgs");
//     }
// }
fn main() {}
