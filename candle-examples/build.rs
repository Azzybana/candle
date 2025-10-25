#![allow(unused)]
use anyhow::{Context, Result};
use std::env;
use std::io::Write;
use std::path::{Path, PathBuf};
mod buildtime_downloader;
use buildtime_downloader::download_model;

fn main() -> Result<()> {
    println!("cargo::rerun-if-changed=build.rs");

    // Download config, tokenizer, and model files from hf at build time.
    // option_env! automatically detects changes in the env var and trigger rebuilds correctly.
    // Example value:
    // CANDLE_BUILDTIME_MODEL_REVISION="sentence-transformers/all-MiniLM-L6-v2:c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
    if let Some(model_rev) = core::option_env!("CANDLE_BUILDTIME_MODEL_REVISION") {
        buildtime_downloader::download_model(model_rev)?;
    }
    Ok(())
}
