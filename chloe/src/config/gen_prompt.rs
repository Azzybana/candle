use std::fs;
use anyhow::Result;
use super::default::ChloeConfig;

impl ChloeConfig {
    pub fn generate_prompt_file(path: &str) -> Result<()> {
        let default_prompt = "Write a Rust function to calculate the factorial of a given number.";
        fs::write(path, default_prompt)?;
        Ok(())
    }

    pub fn generate_chat_prompt_file(path: &str) -> Result<()> {
        let chat_prompt = "You are a helpful AI assistant. Respond to the user's message.";
        fs::write(path, chat_prompt)?;
        Ok(())
    }

    pub fn generate_code_prompt_file(path: &str) -> Result<()> {
        let code_prompt = "You are a code generation assistant. Generate code based on the user's request.";
        fs::write(path, code_prompt)?;
        Ok(())
    }
}