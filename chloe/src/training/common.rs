use crate::config::default::TrainingConfig;
use anyhow::Result;
use rust_tokenizers::tokenizer::Gpt2Tokenizer;
use safetensors::tensor::TensorView;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;
use trash_parallelism::parallel::parallel_map;
use trash_parallelism::serde::serialize_to_file_async;

pub fn create_training_dir(config: &TrainingConfig) -> Result<PathBuf> {
    let training_dir = Path::new(&config.output_gguf)
        .parent()
        .unwrap()
        .join("training");
    fs::create_dir_all(&training_dir)?;
    Ok(training_dir)
}

pub fn load_tokenizer() -> Result<Tokenizer> {
    let tokenizer_path = "data/tokenizer.json";
    Tokenizer::from_file(tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))
}

pub fn load_rust_tokenizer() -> Result<Gpt2Tokenizer> {
    // Assuming the tokenizer is saved as a file, e.g., vocab and merges
    // For GPT2, it might need vocab and merges files
    // Placeholder: adjust based on actual API
    Gpt2Tokenizer::from_file("data/rust_tokenizer.json")
        .map_err(|e| anyhow::anyhow!("Failed to load Rust tokenizer: {}", e))
}

pub fn tokenize_texts(
    tokenizer: &Tokenizer,
    texts: &[String],
    max_len: usize,
) -> Result<(Vec<i64>, Vec<i64>)> {
    let texts_vec = texts.to_vec();
    let results: Vec<Result<(Vec<i64>, Vec<i64>)>> = parallel_map(texts_vec, |text| {
        let encoding = tokenizer
            .encode(text.as_str(), true)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
        let ids: Vec<i64> = encoding
            .get_ids()
            .iter()
            .take(max_len)
            .map(|&x| x as i64)
            .collect();
        let mask: Vec<i64> = vec![1; ids.len()];
        Ok((ids, mask))
    });

    let mut input_ids = Vec::new();
    let mut attention_masks = Vec::new();
    for result in results {
        let (ids, masks) = result?;
        input_ids.extend(ids);
        attention_masks.extend(masks);
    }

    Ok((input_ids, attention_masks))
}

pub async fn save_training_data(
    tensors: HashMap<String, TensorView<'_>>,
    metadata: serde_json::Value,
    training_dir: &Path,
    filename: &str,
    metadata_filename: &str,
) -> Result<()> {
    let safetensors_path = training_dir.join(filename);
    safetensors::serialize_to_file(&tensors, None, &safetensors_path)?;

    let metadata_path = training_dir.join(metadata_filename);
    let path_str = metadata_path.to_str().unwrap();
    serialize_to_file_async(&metadata, path_str)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to save metadata: {}", e))?;

    println!("Training data saved to: {}", training_dir.display());
    Ok(())
}
