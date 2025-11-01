use std::fs;
use anyhow::Result;
use super::default::ChloeConfig;

impl ChloeConfig {
    pub fn generate_prompt_file(path: &str) -> Result<()> {
        let default_prompt = "Write a Rust function to calculate the factorial of a given number.";
        fs::write(path, default_prompt)?;
        Ok(())
    }
}