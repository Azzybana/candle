// pub mod conversion;
pub mod code_trainer;
pub mod text_trainer;
pub mod conversation_trainer;
pub mod reasoning_trainer;
pub mod combinator;
pub mod common;
pub mod filters;

use crate::config::default::ChloeConfig;
use anyhow::Result;

pub async fn run_training(config: &ChloeConfig) -> Result<()> {
    if let Some(training_config) = &config.training {
        // Prepare training data
        text_trainer::prepare_text_training_data(training_config, &training_config.corpus_path).await?;
        println!("Training data preparation completed.");

        // Run training
        text_trainer::train_text_model(training_config).await?;
        println!("Training completed successfully.");

        // Note: ONNX conversion requires additional setup, skipping for now
        // if training_config.source_safetensors.ends_with(".safetensors") {
        //     conversion::convert_safetensors_to_gguf(training_config).await?;
        //     println!("GGUF conversion completed successfully.");
        // } else if training_config.source_safetensors.ends_with(".gguf") {
        //     println!("Source is already GGUF, skipping conversion.");
        // }
    } else {
        println!("No training configuration found.");
    }
    Ok(())
}