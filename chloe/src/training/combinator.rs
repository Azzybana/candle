use anyhow::Result;
use safetensors::tensor::TensorView;
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// Combines two SafeTensors files by concatenating their tensors.
/// Useful for merging multiple small training datasets into larger ones.
pub async fn combine_safetensors_files(file1: &str, file2: &str, output_file: &str) -> Result<()> {
    // Load both SafeTensors files
    let data1 = fs::read(file1)?;
    let data2 = fs::read(file2)?;

    let tensors1 = SafeTensors::deserialize(&data1)?;
    let tensors2 = SafeTensors::deserialize(&data2)?;

    // First pass: collect all data
    let mut data_storage = Vec::new();
    let mut tensor_info = Vec::new(); // (name, dtype, shape, data_index)

    // Get all unique tensor names from both files
    let mut all_names = tensors1.names().into_iter().collect::<std::collections::HashSet<_>>();
    all_names.extend(tensors2.names());

    for name in all_names {
        let tensor1_result = tensors1.tensor(name);
        let tensor2_result = tensors2.tensor(name);

        if let (Ok(tensor1), Ok(tensor2)) = (tensor1_result, tensor2_result) {
            // Both files have this tensor - concatenate them
            let combined_data = combine_tensor_data(&tensor1, &tensor2)?;
            data_storage.push(combined_data);
            let data_index = data_storage.len() - 1;
            let shape = vec![data_storage[data_index].len() / (tensor1.dtype().bitsize() / 8)];
            tensor_info.push((name.to_string(), tensor1.dtype(), shape, data_index));
        } else {
            let tensor_result = tensors1.tensor(name).or_else(|_| tensors2.tensor(name));
            if let Ok(tensor) = tensor_result {
                // Only one file has this tensor - include it as-is
                let tensor_data = tensor.data().to_vec();
                data_storage.push(tensor_data);
                let data_index = data_storage.len() - 1;
                let shape = tensor.shape().to_vec();
                tensor_info.push((name.to_string(), tensor.dtype(), shape, data_index));
            }
        }
    }

    // Second pass: create TensorViews
    let mut combined_tensors = HashMap::new();
    for (name, dtype, shape, data_index) in tensor_info {
        let data_ref = &data_storage[data_index];
        let tensor_view = TensorView::new(dtype, shape, data_ref)?;
        combined_tensors.insert(name, tensor_view);
    }

    // Save combined tensors
    safetensors::serialize_to_file(&combined_tensors, None, Path::new(output_file))?;

    println!("Combined {} and {} into {}", file1, file2, output_file);
    Ok(())
}

/// Combines two tensor data buffers by concatenation.
/// Assumes tensors are 1D (flattened sequences) for training data.
fn combine_tensor_data(tensor1: &safetensors::tensor::TensorView, tensor2: &safetensors::tensor::TensorView) -> Result<Vec<u8>> {
    if tensor1.dtype() != tensor2.dtype() {
        return Err(anyhow::anyhow!("Cannot combine tensors with different dtypes: {:?} vs {:?}", tensor1.dtype(), tensor2.dtype()));
    }

    let mut combined = Vec::new();
    combined.extend_from_slice(tensor1.data());
    combined.extend_from_slice(tensor2.data());

    Ok(combined)
}

/// Combines multiple SafeTensors files sequentially.
/// Useful for batch combining many small datasets.
pub async fn combine_multiple_safetensors_files(files: &[String], output_file: &str) -> Result<()> {
    if files.is_empty() {
        return Err(anyhow::anyhow!("No input files provided"));
    }

    if files.len() == 1 {
        // Just copy the single file
        fs::copy(&files[0], output_file)?;
        println!("Copied single file {} to {}", files[0], output_file);
        return Ok(());
    }

    // Start with the first file
    let mut current_combined = files[0].clone();

    for file in &files[1..] {
        let temp_output = format!("{}.temp", output_file);
        combine_safetensors_files(&current_combined, file, &temp_output).await?;

        // Replace current combined with temp file
        if current_combined != files[0] {
            fs::remove_file(&current_combined)?;
        }
        current_combined = temp_output;
    }

    // Move final result to output location
    fs::rename(&current_combined, output_file)?;
    println!("Combined {} files into {}", files.len(), output_file);

    Ok(())
}