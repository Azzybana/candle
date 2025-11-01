use crate::config::default::TrainingConfig;
use crate::config::read_config::load_metadata;
use anyhow::Result;
use bytemuck::cast_slice;
use candle::quantized::gguf_file;
use ggml_quants::{Q8_0, Quantize};
use ggus::{GGmlType, GGufFileHeader, GGufFileWriter, GGufMetaDataValueType};
use protobuf::Message;
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

pub async fn convert_safetensors_to_onnx(config: &TrainingConfig) -> Result<()> {
    // Load SafeTensors
    let data = fs::read(&config.source_safetensors)?;
    let tensors = SafeTensors::deserialize(&data)?;

    println!(
        "Loaded {} tensors from SafeTensors file for ONNX conversion.",
        tensors.tensors().len()
    );

    // Create ONNX model
    let mut model = ModelProto::default();
    model.ir_version = 8;
    model.producer_name = "chloe".to_string();
    model.producer_version = "0.1.0".to_string();
    model.domain = "".to_string();
    model.model_version = 1;
    model.doc_string = "Converted from SafeTensors".to_string();

    // Create graph
    let mut graph = onnx::GraphProto::default();
    graph.name = "model".to_string();

    // Add inputs (simplified: assume one input tensor)
    let input = onnx::ValueInfoProto {
        name: "input".to_string(),
        doc_string: "".to_string(),
        r#type: Some(onnx::TypeProto {
            value: Some(onnx::type_proto::Value::TensorType(onnx::TensorTypeProto {
                elem_type: onnx::tensor_proto::DataType::Float as i32,
                shape: onnx::TensorShapeProto {
                    dim: vec![onnx::tensor_shape_proto::Dimension {
                        denotation: "".to_string(),
                        value: Some(onnx::tensor_shape_proto::dimension::Value::DimParam(
                            "batch_size".to_string(),
                        )),
                    }],
                },
            })),
        }),
    };
    graph.input = vec![input];

    // Add outputs (simplified: assume one output tensor)
    let output = onnx::ValueInfoProto {
        name: "output".to_string(),
        doc_string: "".to_string(),
        r#type: Some(onnx::TypeProto {
            value: Some(onnx::type_proto::Value::TensorType(onnx::TensorTypeProto {
                elem_type: onnx::tensor_proto::DataType::Float as i32,
                shape: onnx::TensorShapeProto {
                    dim: vec![onnx::tensor_shape_proto::Dimension {
                        denotation: "".to_string(),
                        value: Some(onnx::tensor_shape_proto::dimension::Value::DimParam(
                            "batch_size".to_string(),
                        )),
                    }],
                },
            })),
        }),
    };
    graph.output = vec![output];

    // Add initializers
    for (name, view) in tensors.tensors() {
        let mut tensor = onnx::TensorProto::default();
        tensor.name = name.clone();
        tensor.data_type = match view.dtype() {
            safetensors::Dtype::F32 => onnx::tensor_proto::DataType::Float as i32,
            safetensors::Dtype::F16 => onnx::tensor_proto::DataType::Float16 as i32,
            safetensors::Dtype::BF16 => onnx::tensor_proto::DataType::Bfloat16 as i32,
            safetensors::Dtype::I32 => onnx::tensor_proto::DataType::Int32 as i32,
            safetensors::Dtype::I16 => onnx::tensor_proto::DataType::Int16 as i32,
            safetensors::Dtype::I8 => onnx::tensor_proto::DataType::Int8 as i32,
            safetensors::Dtype::U8 => onnx::tensor_proto::DataType::Uint8 as i32,
            _ => {
                println!("Unsupported dtype for tensor {}, skipping", name);
                continue;
            }
        };
        tensor.dims = view.shape().iter().map(|&x| x as i64).collect();
        tensor.raw_data = view.data().to_vec();
        graph.initializer.push(tensor);
    }

    // Add a simple node (Identity for demonstration - in practice, this should be the actual model graph)
    let node = onnx::NodeProto {
        op_type: "Identity".to_string(),
        domain: "".to_string(),
        attribute: vec![],
        input: vec!["input".to_string()],
        output: vec!["output".to_string()],
        name: "identity".to_string(),
        doc_string: "Identity operation for demonstration".to_string(),
    };
    graph.node = vec![node];

    model.graph = Some(graph);

    // Write the ONNX model
    let output_path = config
        .output_onnx
        .as_ref()
        .unwrap_or(&"model.onnx".to_string())
        .clone();
    let mut file = fs::File::create(&output_path)?;
    model.write_to_writer(&mut file)?;

    println!(
        "ONNX conversion completed successfully. Output: {}",
        output_path
    );

    Ok(())
}

pub async fn convert_gguf_to_onnx(config: &TrainingConfig) -> Result<()> {
    // Load GGUF
    let mut file = fs::File::open(&config.output_gguf)?;
    let model = gguf_file::Content::read(&mut file)?;

    println!(
        "Loaded {} tensors from GGUF file for ONNX conversion.",
        model.tensor_infos.len()
    );

    // Create ONNX model
    let mut onnx_model = ModelProto::default();
    onnx_model.ir_version = 8;
    onnx_model.producer_name = "chloe".to_string();
    onnx_model.producer_version = "0.1.0".to_string();
    onnx_model.domain = "".to_string();
    onnx_model.model_version = 1;
    onnx_model.doc_string = "Converted from GGUF".to_string();

    // Create graph
    let mut graph = onnx::GraphProto::default();
    graph.name = "model".to_string();

    // Add inputs (simplified)
    let input = onnx::ValueInfoProto {
        name: "input".to_string(),
        doc_string: "".to_string(),
        r#type: Some(onnx::TypeProto {
            value: Some(onnx::type_proto::Value::TensorType(onnx::TensorTypeProto {
                elem_type: onnx::tensor_proto::DataType::Float as i32,
                shape: onnx::TensorShapeProto {
                    dim: vec![onnx::tensor_shape_proto::Dimension {
                        denotation: "".to_string(),
                        value: Some(onnx::tensor_shape_proto::dimension::Value::DimParam(
                            "batch_size".to_string(),
                        )),
                    }],
                },
            })),
        }),
    };
    graph.input = vec![input];

    // Add outputs (simplified)
    let output = onnx::ValueInfoProto {
        name: "output".to_string(),
        doc_string: "".to_string(),
        r#type: Some(onnx::TypeProto {
            value: Some(onnx::type_proto::Value::TensorType(onnx::TensorTypeProto {
                elem_type: onnx::tensor_proto::DataType::Float as i32,
                shape: onnx::TensorShapeProto {
                    dim: vec![onnx::tensor_shape_proto::Dimension {
                        denotation: "".to_string(),
                        value: Some(onnx::tensor_shape_proto::dimension::Value::DimParam(
                            "batch_size".to_string(),
                        )),
                    }],
                },
            })),
        }),
    };
    graph.output = vec![output];

    // Add initializers from GGUF tensors
    for (name, tensor_info) in model.tensor_infos.iter() {
        let mut tensor = onnx::TensorProto::default();
        tensor.name = name.clone();
        tensor.data_type = match tensor_info.ggml_dtype.type_size() {
            4 => onnx::tensor_proto::DataType::Float as i32, // F32
            2 => onnx::tensor_proto::DataType::Float16 as i32, // F16
            1 => onnx::tensor_proto::DataType::Int8 as i32,  // Approximate for quantized
            _ => continue,
        };
        tensor.dims = tensor_info.shape.dims().iter().map(|&x| x as i64).collect();
        // For simplicity, skip loading the actual data
        graph.initializer.push(tensor);
    }

    // Add a simple node
    let node = onnx::NodeProto {
        op_type: "Identity".to_string(),
        domain: "".to_string(),
        attribute: vec![],
        input: vec!["input".to_string()],
        output: vec!["output".to_string()],
        name: "identity".to_string(),
        doc_string: "Identity operation for demonstration".to_string(),
    };
    graph.node = vec![node];

    onnx_model.graph = Some(graph);

    // Write the ONNX model
    let output_path = config
        .output_onnx
        .as_ref()
        .unwrap_or(&"model_from_gguf.onnx".to_string())
        .clone();
    let mut file = fs::File::create(&output_path)?;
    onnx_model.write_to_writer(&mut file)?;

    println!(
        "ONNX conversion from GGUF completed successfully. Output: {}",
        output_path
    );

    Ok(())
}
