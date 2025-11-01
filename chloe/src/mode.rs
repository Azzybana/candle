pub mod chat;
pub mod code;
pub mod common;

use crate::config::default::ChloeConfig;
use anyhow::Result;

pub enum Mode {
    Chat,
    Code,
}

pub async fn run_mode(config: &ChloeConfig, mode: Mode) -> Result<()> {
    match mode {
        Mode::Chat => {
            if let Some(chat_config) = &config.chat {
                chat::run_chat(config, chat_config).await?;
            } else {
                println!("No chat configuration found.");
            }
        }
        Mode::Code => {
            if let Some(code_config) = &config.code {
                code::run_code(config, code_config).await?;
            } else {
                println!("No code configuration found.");
            }
        }
    }
    Ok(())
}