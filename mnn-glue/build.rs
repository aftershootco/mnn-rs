use std::path::{Path, PathBuf};
type Result<T, E = Box<dyn std::error::Error>> = std::result::Result<T, E>;
const VENDOR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/vendor");

use cxx_build::CFG;
fn main() -> Result<()> {
    let out_dir = PathBuf::from(std::env::var("OUT_DIR")?);
    let vendor = out_dir.join("vendor");
    cxx_bindings(&vendor)?;
    Ok(())
}

pub fn cxx_bindings(_path: impl AsRef<Path>) -> Result<()> {
    CFG.include_prefix = "mnn-glue";
    let mut builder = cxx_build::bridge("src/shared.rs");
    builder.std("c++14").compile("mnn-cxx-glue");
    Ok(())
}
