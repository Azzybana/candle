use crate::config::default::TrainingConfig;
use crate::training::code_trainer::tokenizer::RustTokenizer;
use crate::training::code_trainer::abstraction::abstract_rust_code;
use crate::training::filters::{
    collect_files_with_extension, filter_files_by_content, is_valid_rust_code,
};
use anyhow::Result;
use std::path::Path;
use trash_parallelism::io::utils::read_file_async;
use trash_parallelism::parallel::advanced::parallel_map_async;
use std::fs::File;
use std::io::Write;
use bytemuck;

// Define a simple protobuf message for batch data (manual implementation)
pub struct BatchData {
    pub tokens: Vec<i32>,
    pub lengths: Vec<i32>,
}

// Orchestration for Rust code training data preparation

pub async fn prepare_efficient_rust_training_data(
    config: &TrainingConfig,
    project_path: &str,
    sp_model_path: &str,
) -> Result<()> {
    let training_dir = Path::new(&config.output_gguf)
        .parent()
        .unwrap()
        .join(".training");
    std::fs::create_dir_all(&training_dir)?;

    // Load tokenizer
    let tokenizer = RustTokenizer::new(sp_model_path)?;

    let code_files = collect_files_with_extension(project_path, "rs");
    let valid_code_files = filter_files_by_content(&code_files, is_valid_rust_code);

    // Parallel abstraction and pre-tokenization
    let all_abstracted: Vec<String> = parallel_map_async(
        valid_code_files,
        |file| async move {
            match read_file_async(&file).await {
                Ok(content) => match abstract_rust_code(&content) {
                    Ok(abs) => abs,
                    Err(_) => Vec::new(),
                },
                Err(_) => Vec::new(),
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

    // Batch and process
    let batch_size = 100;
    for (batch_idx, batch) in all_abstracted.chunks(batch_size).enumerate() {
        let tokenized = tokenizer.tokenize_batch(batch)?;
        save_partial_batch(&tokenized, batch_idx, &training_dir)?;
    }

    Ok(())
}

fn save_partial_batch(tokenized: &[Vec<i32>], batch_idx: usize, training_dir: &Path) -> Result<()> {
    let filename = format!("batch_{}.pb", batch_idx);
    let path = training_dir.join(filename);
    
    // Flatten the tokenized data
    let mut flat_tokens: Vec<i32> = Vec::new();
    let mut lengths: Vec<i32> = Vec::new();
    for ids in tokenized {
        lengths.push(ids.len() as i32);
        flat_tokens.extend(ids);
    }
    
    let mut file = File::create(path)?;
    
    // Write field 1: tokens, repeated int32, packed
    let tokens_bytes = bytemuck::cast_slice(&flat_tokens);
    let len = tokens_bytes.len() as u32;
    file.write_all(&[10])?; // tag 1, wire type 2 (length delimited)
    write_varint(&mut file, len)?;
    file.write_all(tokens_bytes)?;
    
    // Write field 2: lengths, repeated int32, packed
    let lengths_bytes = bytemuck::cast_slice(&lengths);
    let len2 = lengths_bytes.len() as u32;
    file.write_all(&[18])?; // tag 2, wire type 2
    write_varint(&mut file, len2)?;
    file.write_all(lengths_bytes)?;
    
    Ok(())
}

fn write_varint(writer: &mut File, mut value: u32) -> Result<()> {
    while value >= 0x80 {
        writer.write_all(&[(value & 0x7F) as u8 | 0x80])?;
        value >>= 7;
    }
    writer.write_all(&[value as u8])?;
    Ok(())
}
