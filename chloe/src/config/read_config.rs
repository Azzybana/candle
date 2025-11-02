use super::default::ChloeConfig;
use anyhow::Result;
use serde_json;
use std::collections::HashMap;
use std::fs;
use toml;
use trash_parallelism::io::utils::read_file_async;

impl ChloeConfig {
    pub async fn load_from_file(path: &str) -> Result<Self> {
        let content = read_file_async(path).await?;
        let config: ChloeConfig = toml::from_str(&content)?;
        Ok(config)
    }

    pub async fn find_config() -> Option<String> {
        let search_paths = ["", "data/", ".config/", "config/", "model/"];
        for dir in &search_paths {
            let path = format!("{}config.toml", dir);
            if std::path::Path::new(&path).exists() && Self::load_from_file(&path).await.is_ok() {
                return Some(path);
            }
        }
        None
    }
}

pub fn load_vocab(tokenizer_path: &str) -> Result<HashMap<String, u32>> {
    let content = fs::read_to_string(tokenizer_path)?;
    let json: serde_json::Value = serde_json::from_str(&content)?;
    let vocab_obj = json["model"]["vocab"]
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("No vocab in tokenizer JSON"))?;
    let mut vocab = HashMap::new();
    for (k, v) in vocab_obj {
        if let Some(id) = v.as_u64() {
            vocab.insert(k.clone(), id as u32);
        }
    }
    Ok(vocab)
}

#[allow(dead_code)]
pub fn load_metadata(metadata_path: &str) -> Result<serde_json::Value> {
    let content = fs::read_to_string(metadata_path)?;
    let json: serde_json::Value = serde_json::from_str(&content)?;
    Ok(json)
}
