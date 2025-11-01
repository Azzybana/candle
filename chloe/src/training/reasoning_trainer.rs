use crate::config::default::TrainingConfig;
use anyhow::Result;
use safetensors::tensor::TensorView;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use tokenizers::Tokenizer;
use walkdir::WalkDir;

#[derive(Deserialize, Serialize)]
struct ReasoningProblem {
    problem: String,
    solution: String,
}

pub async fn prepare_reasoning_training_data(config: &TrainingConfig, problems_path: &str) -> Result<()> {
    // Create training subfolder
    let training_dir = Path::new(&config.output_gguf).parent().unwrap().join("training");
    fs::create_dir_all(&training_dir)?;

    // Load tokenizer
    let tokenizer_path = "data/tokenizer.json";
    let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

    // Scan for JSON files
    let mut problems = Vec::new();
    for entry in WalkDir::new(problems_path).into_iter().filter_map(|e| e.ok()) {
        if entry.path().extension().and_then(|s| s.to_str()) == Some("json") {
            if let Ok(content) = fs::read_to_string(entry.path()) {
                if let Ok::<Vec<ReasoningProblem>, _>(data) = serde_json::from_str(&content) {
                    problems.extend(data);
                }
            }
        }
    }

    // Tokenize
    let mut input_ids = Vec::new();
    let mut attention_masks = Vec::new();
    let max_len = 512;

    for prob in &problems {
        let text = format!("Problem: {}\nSolution: {}", prob.problem, prob.solution);
        let encoding = tokenizer.encode(text.as_str(), true).map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
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

    let safetensors_path = training_dir.join("reasoning_training_data.safetensors");
    safetensors::serialize_to_file(&tensors, None, &safetensors_path)?;

    // Save metadata
    let metadata = serde_json::json!({
        "num_problems": problems.len(),
        "max_len": max_len,
        "tokenizer": tokenizer_path,
        "problems_path": problems_path
    });
    let metadata_path = training_dir.join("reasoning_metadata.json");
    fs::write(&metadata_path, serde_json::to_string_pretty(&metadata)?)?;

    println!("Reasoning training data prepared in: {}", training_dir.display());
    Ok(())
}