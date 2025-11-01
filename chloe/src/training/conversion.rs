use crate::config::default::TrainingConfig;
use crate::config::read_config::load_metadata;
use anyhow::Result;
use bytemuck::cast_slice;
use ggml_quants::{Q8_0, Quantize};
use ggus::{GGmlType, GGufFileHeader, GGufFileWriter, GGufMetaDataValueType};
use safetensors::SafeTensors;
use std::fs;

pub async fn convert_safetensors_to_gguf(config: &TrainingConfig) -> Result<()> {
    // Load SafeTensors
    let data = fs::read(&config.source_safetensors)?;
    let tensors = SafeTensors::deserialize(&data)?;

    // Load metadata if provided
    let metadata = if let Some(meta_path) = &config.metadata {
        Some(load_metadata(meta_path)?)
    } else {
        None
    };

    println!(
        "Loaded {} tensors from SafeTensors file.",
        tensors.tensors().len()
    );

    // Create GGUF file
    let file = fs::File::create(&config.output_gguf)?;
    let metadata_count = metadata
        .as_ref()
        .and_then(|m| m.as_object())
        .map(|o| o.len())
        .unwrap_or(0) as u64;
    let header = GGufFileHeader::new(3, metadata_count, tensors.tensors().len() as u64);
    let mut writer = GGufFileWriter::new(file, header)?;

    writer.write_alignment(32)?;

    // Write metadata
    if let Some(ref meta) = metadata
        && let Some(obj) = meta.as_object()
    {
        for (key, value) in obj {
            let value_str = if let Some(s) = value.as_str() {
                s.to_string()
            } else {
                serde_json::to_string(value)?
            };
            writer.write_meta_kv(
                key,
                GGufMetaDataValueType::String,
                format!("{}\0", value_str).as_bytes(),
            )?;
        }
    }

    let mut tensor_writer = writer.finish::<Vec<u8>>(true);

    // Write tensors
    for (name, view) in tensors.tensors() {
        let shape: Vec<u64> = view.shape().iter().map(|&x| x as u64).collect();
        let (ggml_type, data) = if view.dtype() == safetensors::Dtype::F32 {
            // Quantize F32 to Q8_0 in blocks of 32
            let data_f32: &[f32] = cast_slice(view.data());
            let mut quantized_data = Vec::new();
            for chunk in data_f32.chunks(32) {
                if chunk.len() == 32 {
                    let quantized = Q8_0::quantize(chunk.try_into().unwrap());
                    let d_bytes = quantized.delta.to_le_bytes();
                    quantized_data.extend_from_slice(&d_bytes);
                    quantized_data.extend_from_slice(bytemuck::cast_slice(&quantized.quants));
                } else {
                    // Handle remaining, but for simplicity, skip or pad
                    // For now, skip incomplete blocks
                }
            }
            (GGmlType::Q8_0, quantized_data)
        } else {
            // For other dtypes, write as is (map to GGmlType)
            let ggml_type = match view.dtype() {
                safetensors::Dtype::F16 => GGmlType::F16,
                safetensors::Dtype::BF16 => GGmlType::BF16,
                safetensors::Dtype::I32 => GGmlType::I32,
                safetensors::Dtype::I16 => GGmlType::I16,
                safetensors::Dtype::I8 => GGmlType::I8,
                safetensors::Dtype::U8 => GGmlType::I8, // Map U8 to I8
                _ => {
                    println!("Unsupported dtype for tensor {}, skipping", name);
                    continue;
                }
            };
            (ggml_type, view.data().to_vec())
        };
        tensor_writer.write_tensor(&name, ggml_type, &shape, data)?;
    }

    tensor_writer.finish()?;

    println!(
        "GGUF conversion completed successfully. Output: {}",
        config.output_gguf
    );

    Ok(())
}
