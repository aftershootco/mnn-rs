use anyhow::*;
use clap::Parser;
use std::path::{Path, PathBuf};
#[derive(Debug, clap::Parser)]
pub struct Cli {
    #[clap(short, long)]
    model: PathBuf,
}
pub fn main() -> Result<()> {
    let cli = Cli::parse();
    let mut llm = mnn::llm::Llm::from_file(cli.model)?;
    llm.load()?;
    llm.chat()?;
    Ok(())
}
