use crate::config::default::TrainingConfig;
use anyhow::Result;

mod rust;
mod tokenizer;
mod abstraction;
mod simd;

#[allow(dead_code)]
pub async fn prepare_code_training_data(config: &TrainingConfig, project_path: &str) -> Result<()> {
    // Call the efficient trainer from rust.rs
    crate::training::code_trainer::rust::prepare_efficient_rust_training_data(
        config,
        project_path,
        "data/rust_sp.model",
    )
    .await
}
