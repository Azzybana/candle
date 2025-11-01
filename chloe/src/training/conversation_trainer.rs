use crate::config::default::TrainingConfig;
use crate::training::common::{create_training_dir, load_tokenizer, tokenize_texts, save_training_data};
use crate::training::filters::{collect_files_with_extension, filter_files_by_content, is_valid_json, deduplicate_text_samples};
use anyhow::Result;
use safetensors::tensor::TensorView;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;

#[derive(Deserialize, Serialize)]
struct Conversation {
    user: String,
    assistant: String,
}

pub async fn prepare_conversation_training_data(config: &TrainingConfig, conversations_path: &str) -> Result<()> {
    let training_dir = create_training_dir(config)?;
    let tokenizer = load_tokenizer()?;

    let json_files = collect_files_with_extension(conversations_path, "json");
    let valid_json_files = filter_files_by_content(&json_files, is_valid_json);

    let mut conversations = Vec::new();
    for file in valid_json_files {
        if let Ok(content) = fs::read_to_string(&file) {
            if let Ok::<Vec<Conversation>, _>(data) = serde_json::from_str(&content) {
                conversations.extend(data);
            }
        }
    }

    let texts: Vec<String> = conversations.iter().map(|conv| format!("<|user|>{}<|assistant|>{}", conv.user, conv.assistant)).collect();

    let original_count = texts.len();
    // Deduplicate conversation texts
    let deduplicated_texts = deduplicate_text_samples(texts);
    println!("Generated {} conversation samples, {} after deduplication", original_count, deduplicated_texts.len());

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
        "conversations_path": conversations_path
    });

    save_training_data(tensors, metadata, &training_dir, "conversation_training_data.safetensors", "conversation_metadata.json").await?;

    Ok(())
}