use std::fs;
use std::collections::HashMap;
use anyhow::Result;
use serde_json;
use toml;
use super::default::ChloeConfig;

impl ChloeConfig {
    pub fn load_from_file(path: &str) -> Result<Self> {
        let content = fs::read_to_string(path)?;
        let config: ChloeConfig = toml::from_str(&content)?;
        Ok(config)
    }

    pub fn find_config() -> Option<String> {
        let search_paths = ["", "data/", ".config/", "config/", "model/"];
        for dir in &search_paths {
            let path = format!("{}config.toml", dir);
            if std::path::Path::new(&path).exists()
                && Self::load_from_file(&path).is_ok() {
                return Some(path);
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