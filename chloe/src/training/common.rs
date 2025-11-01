use crate::config::default::TrainingConfig;
use anyhow::Result;
use safetensors::tensor::TensorView;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;

pub fn create_training_dir(config: &TrainingConfig) -> Result<PathBuf> {
    let training_dir = Path::new(&config.output_gguf).parent().unwrap().join("training");
    fs::create_dir_all(&training_dir)?;
    Ok(training_dir)
}

pub fn load_tokenizer() -> Result<Tokenizer> {
    let tokenizer_path = "data/tokenizer.json";
    Tokenizer::from_file(tokenizer_path).map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))
}

pub fn tokenize_texts(tokenizer: &Tokenizer, texts: &[String], max_len: usize) -> Result<(Vec<i64>, Vec<i64>)> {
    let mut input_ids = Vec::new();
    let mut attention_masks = Vec::new();

    for text in texts {
        let encoding = tokenizer.encode(text.as_str(), true).map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
        let ids: Vec<i64> = encoding.get_ids().iter().take(max_len).map(|&x| x as i64).collect();
        let mask: Vec<i64> = vec![1; ids.len()];
        input_ids.extend(ids);
        attention_masks.extend(mask);
    }

    Ok((input_ids, attention_masks))
}

pub fn save_training_data(tensors: HashMap<String, TensorView>, metadata: serde_json::Value, training_dir: &Path, filename: &str, metadata_filename: &str) -> Result<()> {
    let safetensors_path = training_dir.join(filename);
    safetensors::serialize_to_file(&tensors, None, &safetensors_path)?;

    let metadata_path = training_dir.join(metadata_filename);
    fs::write(&metadata_path, serde_json::to_string_pretty(&metadata)?)?;

    println!("Training data saved to: {}", training_dir.display());
    Ok(())
}