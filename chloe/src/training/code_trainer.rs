use crate::config::default::TrainingConfig;
use anyhow::Result;
use safetensors::tensor::TensorView;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use syn::{visit::Visit, File, ItemFn, ItemStruct};
use tokenizers::Tokenizer;
use walkdir::WalkDir;

pub async fn prepare_code_training_data(config: &TrainingConfig, project_path: &str) -> Result<()> {
    // Create training subfolder
    let training_dir = Path::new(&config.output_gguf).parent().unwrap().join("training");
    fs::create_dir_all(&training_dir)?;

    // Load tokenizer (assume it's in data/)
    let tokenizer_path = "data/tokenizer.json";
    let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

    // Scan for .rs files
    let mut code_samples = Vec::new();
    for entry in WalkDir::new(project_path).into_iter().filter_map(|e| e.ok()) {
        if entry.path().extension().and_then(|s| s.to_str()) == Some("rs") {
            if let Ok(content) = fs::read_to_string(entry.path()) {
                if let Ok(abstracted) = abstract_rust_code(&content) {
                    code_samples.push(abstracted);
                }
            }
        }
    }

    // Tokenize
    let mut input_ids = Vec::new();
    let mut attention_masks = Vec::new();
    let max_len = 512; // Example max length
    let num_samples = code_samples.len();

    for sample in code_samples {
        let encoding = tokenizer.encode(sample, true).map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
        let ids: Vec<i64> = encoding.get_ids().iter().take(max_len).map(|&x| x as i64).collect();
        let mask: Vec<i64> = vec![1; ids.len()];
        input_ids.extend(ids);
        attention_masks.extend(mask);
    }

    // Pad or truncate to fixed size if needed, but for simplicity, save as is
    // For safetensors, need tensors
    let input_ids_tensor = TensorView::new(safetensors::Dtype::I64, vec![input_ids.len()], bytemuck::cast_slice(&input_ids))?;
    let attention_masks_tensor = TensorView::new(safetensors::Dtype::I64, vec![attention_masks.len()], bytemuck::cast_slice(&attention_masks))?;

    let mut tensors = HashMap::new();
    tensors.insert("input_ids".to_string(), input_ids_tensor);
    tensors.insert("attention_mask".to_string(), attention_masks_tensor);

    // Save safetensors
    let safetensors_path = training_dir.join("training_data.safetensors");
    safetensors::serialize_to_file(&tensors, None, &safetensors_path)?;

    // Save metadata JSON
    let metadata = serde_json::json!({
        "num_samples": num_samples,
        "max_len": max_len,
        "tokenizer": tokenizer_path,
        "project_path": project_path
    });
    let metadata_path = training_dir.join("metadata.json");
    fs::write(&metadata_path, serde_json::to_string_pretty(&metadata)?)?;

    println!("Training data prepared in: {}", training_dir.display());
    Ok(())
}

fn abstract_rust_code(code: &str) -> Result<String> {
    let syntax_tree: File = syn::parse_str(code)?;
    let mut visitor = CodeAbstractor::new();
    visitor.visit_file(&syntax_tree);
    Ok(visitor.abstracted)
}

struct CodeAbstractor {
    abstracted: String,
}

impl CodeAbstractor {
    fn new() -> Self {
        Self {
            abstracted: String::new(),
        }
    }
}

impl<'ast> Visit<'ast> for CodeAbstractor {
    fn visit_item_fn(&mut self, node: &'ast ItemFn) {
        self.abstracted.push_str(&format!("fn {} (", node.sig.ident));
        for param in &node.sig.inputs {
            match param {
                syn::FnArg::Receiver(_) => self.abstracted.push_str("self, "),
                syn::FnArg::Typed(pat_type) => {
                    if let syn::Pat::Ident(pat_ident) = &*pat_type.pat {
                        self.abstracted.push_str(&format!("{}: {}, ", pat_ident.ident, "Type"));
                    }
                }
            }
        }
        self.abstracted.push_str(") -> ReturnType {\n");
        // Abstract body, e.g., count statements
        let stmt_count = node.block.stmts.len();
        self.abstracted.push_str(&format!("  // {} statements\n", stmt_count));
        self.abstracted.push_str("}\n\n");
    }

    fn visit_item_struct(&mut self, node: &'ast ItemStruct) {
        self.abstracted.push_str(&format!("struct {} {{\n", node.ident));
        for field in &node.fields {
            if let Some(ident) = &field.ident {
                self.abstracted.push_str(&format!("  {}: Type,\n", ident));
            }
        }
        self.abstracted.push_str("}\n\n");
    }

    // Add more visit methods for other items as needed
}