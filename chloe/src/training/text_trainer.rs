use crate::config::default::TrainingConfig;
use anyhow::Result;
use safetensors::tensor::TensorView;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use tokenizers::Tokenizer;
use walkdir::WalkDir;

pub async fn prepare_text_training_data(config: &TrainingConfig, corpus_path: &str) -> Result<()> {
    // Create training subfolder
    let training_dir = Path::new(&config.output_gguf).parent().unwrap().join("training");
    fs::create_dir_all(&training_dir)?;

    // Load tokenizer
    let tokenizer_path = "data/tokenizer.json";
    let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

    // Scan for text files
    let mut text_samples = Vec::new();
    for entry in WalkDir::new(corpus_path).into_iter().filter_map(|e| e.ok()) {
        if entry.path().extension().and_then(|s| s.to_str()) == Some("txt") || entry.path().extension().and_then(|s| s.to_str()) == Some("md") {
            if let Ok(content) = fs::read_to_string(entry.path()) {
                text_samples.push(content);
            }
        }
    }

    // Tokenize
    let mut input_ids = Vec::new();
    let mut attention_masks = Vec::new();
    let max_len = 512;

    for sample in &text_samples {
        let encoding = tokenizer.encode(sample.as_str(), true).map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
        let ids: Vec<i64> = encoding.get_ids().iter().take(max_len).map(|&x| x as i64).collect();
        let mask: Vec<i64> = vec![1; ids.len()];
        input_ids.extend(ids);
        attention_masks.extend(mask);
    }

    // Save tensors
    let input_ids_tensor = TensorView::new(safetensors::Dtype::I64, vec![input_ids.len()], bytemuck::cast_slice(&input_ids))?;
    let attention_masks_tensor = TensorView::new(safetensors::Dtype::I64, vec![attention_masks.len()], bytemuck::cast_slice(&attention_masks))?;

    let mut tensors = HashMap::new();
    tensors.insert("input_ids".to_string(), input_ids_tensor);
    tensors.insert("attention_mask".to_string(), attention_masks_tensor);

    let safetensors_path = training_dir.join("text_training_data.safetensors");
    safetensors::serialize_to_file(&tensors, None, &safetensors_path)?;

    // Save metadata
    let metadata = serde_json::json!({
        "num_samples": text_samples.len(),
        "max_len": max_len,
        "tokenizer": tokenizer_path,
        "corpus_path": corpus_path
    });
    let metadata_path = training_dir.join("text_metadata.json");
    fs::write(&metadata_path, serde_json::to_string_pretty(&metadata)?)?;

    println!("Text training data prepared in: {}", training_dir.display());
    Ok(())
}