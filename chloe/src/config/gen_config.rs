use super::default::ChloeConfig;
use anyhow::Result;
use std::fs;

impl ChloeConfig {
    pub fn generate_config_file(path: &str) -> Result<()> {
        let config = Self::default_config();
        let toml_string = toml::to_string(&config)?;
        fs::write(path, toml_string)?;
        // Create prompt.md in the same directory as the config
        let config_dir = std::path::Path::new(path)
            .parent()
            .unwrap_or(std::path::Path::new("."));
        let prompt_path = config_dir.join("prompt.md");
        if !prompt_path.exists() {
            Self::generate_prompt_file(prompt_path.to_str().unwrap())?;
        }
        Ok(())
    }
}
