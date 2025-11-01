use crate::config::default::{ChloeConfig, ChatConfig};
use crate::mode::common::prepare_config_for_mode;
use anyhow::Result;

pub async fn run_chat(config: &ChloeConfig, chat_config: &ChatConfig) -> Result<()> {
    let effective_config = prepare_config_for_mode(&config.chloe, &chat_config.prompt);
    // TODO: Implement inference logic for chat mode
    println!("Running chat mode with prompt: {}", effective_config.prompt);
    // For now, placeholder
    Ok(())
}