use crate::config::default::TrainingConfig;
use crate::training::common::{create_training_dir, load_tokenizer, tokenize_texts, save_training_data};
use crate::training::filters::{collect_files_with_extension, filter_files_by_content, is_valid_json, deduplicate_text_samples};
use anyhow::Result;
use safetensors::tensor::TensorView;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use trash_parallelism::parallel::advanced::parallel_map_async;

#[derive(Deserialize, Serialize)]
#[allow(dead_code)]
struct ReasoningProblem {
    problem: String,
    solution: String,
}

#[allow(dead_code)]
pub async fn prepare_reasoning_training_data(config: &TrainingConfig, problems_path: &str) -> Result<()> {
    let training_dir = create_training_dir(config)?;
    let tokenizer = load_tokenizer()?;

    let json_files = collect_files_with_extension(problems_path, "json");
    let valid_json_files = filter_files_by_content(&json_files, is_valid_json);

    // Process JSON files in parallel
    let problems: Vec<ReasoningProblem> = parallel_map_async(
        valid_json_files,
        |file| async move {
            if let Ok(content) = fs::read_to_string(&file) {
                serde_json::from_str::<Vec<ReasoningProblem>>(&content).unwrap_or_default()
            } else {
                Vec::new()
            }
        },
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
            .min(8),
    )
    .await
    .into_iter()
    .flatten()
    .collect();

    let texts: Vec<String> = problems.iter().map(|prob| format!("Problem: {}\nSolution: {}", prob.problem, prob.solution)).collect();

    let original_count = texts.len();
    // Deduplicate reasoning texts
    let deduplicated_texts = deduplicate_text_samples(texts);
    println!("Generated {} reasoning samples, {} after deduplication", original_count, deduplicated_texts.len());

    let (input_ids, attention_masks) = tokenize_texts(&tokenizer, &deduplicated_texts, 512)?;

    let input_ids_tensor = TensorView::new(safetensors::Dtype::I64, vec![input_ids.len()], bytemuck::cast_slice(&input_ids))?;
    let attention_masks_tensor = TensorView::new(safetensors::Dtype::I64, vec![attention_masks.len()], bytemuck::cast_slice(&attention_masks))?;

    let mut tensors = HashMap::new();
    tensors.insert("input_ids".to_string(), input_ids_tensor);
    tensors.insert("attention_mask".to_string(), attention_masks_tensor);

    let metadata = serde_json::json!({
        "num_samples": deduplicated_texts.len(),
        "max_len": 512,
        "tokenizer": "data/tokenizer.json",
        "problems_path": problems_path
    });

    save_training_data(tensors, metadata, &training_dir, "reasoning_training_data.safetensors", "reasoning_metadata.json").await?;

    Ok(())
}