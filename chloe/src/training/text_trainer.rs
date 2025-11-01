use crate::config::default::TrainingConfig;
use crate::training::common::{create_training_dir, load_tokenizer, tokenize_texts, save_training_data};
use crate::training::filters::collect_files_with_extension;
use anyhow::Result;
use std::collections::HashMap;
use std::fs;
use candle::{DType, Device, Tensor};
use candle_nn::{loss, AdamW, Embedding, Linear, Module, Optimizer, VarMap};
use safetensors;
use safetensors::tensor::TensorView;

pub async fn prepare_text_training_data(config: &TrainingConfig, corpus_path: &str) -> Result<()> {
    let training_dir = create_training_dir(config)?;
    let tokenizer = load_tokenizer()?;

    let text_files = collect_files_with_extension(corpus_path, "txt");
    let mut text_samples = Vec::new();
    for file in text_files {
        if let Ok(content) = fs::read_to_string(&file) {
            text_samples.push(content);
        }
    }

    let (input_ids, attention_masks) = tokenize_texts(&tokenizer, &text_samples, 512)?;

    let input_ids_tensor = TensorView::new(safetensors::Dtype::I64, vec![input_ids.len()], bytemuck::cast_slice(&input_ids))?;
    let attention_masks_tensor = TensorView::new(safetensors::Dtype::I64, vec![attention_masks.len()], bytemuck::cast_slice(&attention_masks))?;

    let mut tensors = HashMap::new();
    tensors.insert("input_ids".to_string(), input_ids_tensor);
    tensors.insert("attention_mask".to_string(), attention_masks_tensor);

    let metadata = serde_json::json!({
        "num_samples": text_samples.len(),
        "max_len": 512,
        "tokenizer": "data/tokenizer.json",
        "corpus_path": corpus_path
    });

    save_training_data(tensors, metadata, &training_dir, "text_training_data.safetensors", "text_metadata.json")?;

    Ok(())
}

pub async fn train_text_model(config: &TrainingConfig) -> Result<()> {
    let device = Device::Cpu; // CPU-only as requested

    // Load prepared data
    let training_dir = create_training_dir(config)?;
    let data_path = training_dir.join("text_training_data.safetensors");
    let data = fs::read(&data_path)?;
    let tensors = safetensors::SafeTensors::deserialize(&data)?;

    let input_ids = tensors.tensor("input_ids")?;
    let attention_mask = tensors.tensor("attention_mask")?;

    // Convert to Candle tensors
    let input_ids = Tensor::from_raw_buffer(
        input_ids.data(),
        DType::I64,
        input_ids.shape(),
        &device,
    )?;
    let attention_mask = Tensor::from_raw_buffer(
        attention_mask.data(),
        DType::I64,
        attention_mask.shape(),
        &device,
    )?;

    // Simple model: Embedding + Linear for text classification (example)
    let vocab_size = 32000; // Assume tokenizer vocab size
    let hidden_size = 768;
    let num_classes = 2; // Binary classification example

    let varmap = VarMap::new();
    let embedding: Embedding = Embedding::new(varmap.get((vocab_size, hidden_size), "embed", candle_nn::Init::Randn { mean: 0.0, stdev: 0.02 }, DType::F32, &device)?, hidden_size);
    let linear: Linear = Linear::new(varmap.get((hidden_size, num_classes), "linear", candle_nn::Init::Randn { mean: 0.0, stdev: 0.02 }, DType::F32, &device)?, None);

    // Dummy labels (for demonstration - in practice, load real labels)
    let batch_size = 4;
    let seq_len = 512;
    let num_samples = input_ids.dim(0)? / seq_len;
    let labels = Tensor::zeros((num_samples, num_classes), DType::F32, &device)?;

    // Training loop
    let mut optimizer = AdamW::new_lr(varmap.all_vars(), 1e-3)?;
    let num_epochs = 5;

    for epoch in 0..num_epochs {
        let mut total_loss = 0.0;
        let mut num_batches = 0;

        // Simple batching (in practice, use proper data loader)
        for i in (0..num_samples).step_by(batch_size) {
            let end = (i + batch_size).min(num_samples);
            let batch_input = input_ids.narrow(0, i * seq_len, (end - i) * seq_len)?;
            let batch_labels = labels.narrow(0, i, end - i)?;

            // Forward pass
            let embedded = embedding.forward(&batch_input)?;
            let pooled = embedded.mean(1)?; // Simple pooling
            let logits = linear.forward(&pooled)?;
            let loss = loss::cross_entropy(&logits, &batch_labels)?;

            // Backward pass
            optimizer.backward_step(&loss)?;

            total_loss += loss.to_scalar::<f32>()?;
            num_batches += 1;
        }

        println!("Epoch {}: Average loss = {:.4}", epoch + 1, total_loss / num_batches as f32);
    }

    // Save trained model using Candle's serialization (SafeTensors)
    let model_path = training_dir.join("trained_text_model.safetensors");
    let tensors_to_save = HashMap::from([
        ("embed.weight".to_string(), embedding.embeddings().clone()),
        ("linear.weight".to_string(), linear.weight().clone()),
        ("linear.bias".to_string(), linear.bias().unwrap().clone()),
    ]);
    safetensors::serialize_to_file(&tensors_to_save, None, &model_path)?;

    println!("Trained model saved to: {}", model_path.display());
    Ok(())
}

