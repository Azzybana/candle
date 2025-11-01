use crate::config::default::{ChloeConfig, CodeConfig};
use crate::mode::common::prepare_config_for_mode;
use anyhow::Result;

pub async fn run_code(config: &ChloeConfig, code_config: &CodeConfig) -> Result<()> {
    let effective_config = prepare_config_for_mode(&config.chloe, &code_config.prompt);
    // TODO: Implement inference logic for code mode
    println!("Running code mode with prompt: {}", effective_config.prompt);
    // For now, placeholder
    Ok(())
}