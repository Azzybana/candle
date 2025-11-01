use serde::{Deserialize, Serialize};
use std::fs;
use std::collections::HashMap;
use anyhow::Result;

#[derive(Deserialize, Serialize, Debug)]
pub struct ChloeConfig {
    pub chloe: Config,
}

#[derive(Deserialize, Serialize, Debug)]
pub struct Config {
    /// Path to the model file (e.g., .gguf, .safetensors)
    pub model: String,
    /// Path to the tokenizer file (e.g., .json)
    pub tokenizer: String,
    /// Path to the prompt file (e.g., .md, .txt)
    pub prompt: String,
}

impl ChloeConfig {
    pub fn load_from_file(path: &str) -> Result<Self> {
        let content = fs::read_to_string(path)?;
        let config: ChloeConfig = toml::from_str(&content)?;
        Ok(config)
    }

    pub fn default_config() -> Self {
        ChloeConfig {
            chloe: Config {
                model: "model.gguf".to_string(),
                tokenizer: "tokenizer.json".to_string(),
                prompt: "prompt.md".to_string(),
            },
        }
    }

    pub fn generate_config_file(path: &str) -> Result<()> {
        let config = Self::default_config();
        let toml_string = toml::to_string(&config)?;
        fs::write(path, toml_string)?;
        // Create prompt.md in the same directory as the config
        let config_dir = std::path::Path::new(path).parent().unwrap_or(std::path::Path::new("."));
        let prompt_path = config_dir.join("prompt.md");
        if !prompt_path.exists() {
            Self::generate_prompt_file(prompt_path.to_str().unwrap())?;
        }
        Ok(())
    }

    pub fn generate_prompt_file(path: &str) -> Result<()> {
        let default_prompt = "Write a Rust function to calculate the factorial of a given number.";
        fs::write(path, default_prompt)?;
        Ok(())
    }

    pub fn find_config() -> Option<String> {
        let search_paths = ["", "data/", ".config/", "config/", "model/"];
        for dir in &search_paths {
            let path = format!("{}config.toml", dir);
            if std::path::Path::new(&path).exists() {
                if Self::load_from_file(&path).is_ok() {
                    return Some(path);
                }
            }
        }
        None
    }
}

pub fn load_vocab(tokenizer_path: &str) -> Result<HashMap<String, u32>> {
    let content = fs::read_to_string(tokenizer_path)?;
    let json: serde_json::Value = serde_json::from_str(&content)?;
    let vocab_obj = json["model"]["vocab"].as_object().ok_or_else(|| anyhow::anyhow!("No vocab in tokenizer JSON"))?;
    let mut vocab = HashMap::new();
    for (k, v) in vocab_obj {
        if let Some(id) = v.as_u64() {
            vocab.insert(k.clone(), id as u32);
        }
    }
    Ok(vocab)
}