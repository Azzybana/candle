pub mod conversion;
pub mod code_trainer;
pub mod text_trainer;
pub mod conversation_trainer;
pub mod reasoning_trainer;
pub mod combinator;

use crate::config::default::ChloeConfig;
use anyhow::Result;

pub async fn run_training(config: &ChloeConfig) -> Result<()> {
    if let Some(training_config) = &config.training {
        conversion::convert_safetensors_to_gguf(training_config).await?;
        println!("Conversion completed successfully.");
    } else {
        println!("No training configuration found.");
    }
    Ok(())
}