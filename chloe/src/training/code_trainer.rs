use crate::config::default::TrainingConfig;
use crate::training::common::{
    create_training_dir, load_tokenizer, save_training_data, tokenize_texts,
};
use crate::training::filters::{collect_files_with_extension, filter_files_by_content, is_valid_rust_code, deduplicate_text_samples};
use trash_parallelism::sys::Timer;
use trash_parallelism::common::utils::AtomicCounter;
use anyhow::Result;
use safetensors::tensor::TensorView;
use std::collections::HashMap;
use syn::{File, ItemFn, ItemStruct, visit::Visit};
use trash_parallelism::io::utils::read_file_async;
use trash_parallelism::parallel::advanced::parallel_map_async;

#[allow(dead_code)]
pub async fn prepare_code_training_data(config: &TrainingConfig, project_path: &str) -> Result<()> {
    let _overall_timer = Timer::new("prepare_code_training_data");
    let _processed_counter = AtomicCounter::new();
    let training_dir = create_training_dir(config)?;
    let tokenizer = load_tokenizer()?;

    let code_files = collect_files_with_extension(project_path, "rs");
    let valid_code_files = filter_files_by_content(&code_files, is_valid_rust_code);
    let total_files = valid_code_files.len();
    println!("Found {} valid Rust files to process", total_files);

    // Process files in parallel using parallel_map_async
    let all_code_samples: Vec<String> = parallel_map_async(
        valid_code_files,
        |file| async move {
            match read_file_async(&file).await {
                Ok(content) => abstract_rust_code(&content).unwrap_or_default(),
                Err(_) => String::new(),
            }
        },
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
            .min(8), // Use up to 8 workers
    )
    .await;

    println!("Successfully processed {} code samples", all_code_samples.len());

    // Deduplicate samples to reduce training data size and improve quality
    let deduplicated_samples = deduplicate_text_samples(all_code_samples);
    println!("After deduplication: {} unique code samples", deduplicated_samples.len());

    let (input_ids, attention_masks) = tokenize_texts(&tokenizer, &deduplicated_samples, 512)?;

    let input_ids_tensor = TensorView::new(
        safetensors::Dtype::I64,
        vec![input_ids.len()],
        bytemuck::cast_slice(&input_ids),
    )?;
    let attention_masks_tensor = TensorView::new(
        safetensors::Dtype::I64,
        vec![attention_masks.len()],
        bytemuck::cast_slice(&attention_masks),
    )?;

    let mut tensors = HashMap::new();
    tensors.insert("input_ids".to_string(), input_ids_tensor);
    tensors.insert("attention_mask".to_string(), attention_masks_tensor);

    let metadata = serde_json::json!({
        "num_samples": deduplicated_samples.len(),
        "max_len": 512,
        "tokenizer": "data/tokenizer.json",
        "project_path": project_path
    });

    save_training_data(
        tensors,
        metadata,
        &training_dir,
        "training_data.safetensors",
        "metadata.json",
    ).await?;

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
        self.abstracted
            .push_str(&format!("fn {} (", node.sig.ident));
        for param in &node.sig.inputs {
            match param {
                syn::FnArg::Receiver(_) => self.abstracted.push_str("self, "),
                syn::FnArg::Typed(pat_type) => {
                    if let syn::Pat::Ident(pat_ident) = &*pat_type.pat {
                        self.abstracted
                            .push_str(&format!("{}: {}, ", pat_ident.ident, "Type"));
                    }
                }
            }
        }
        self.abstracted.push_str(") -> ReturnType {\n");
        // Abstract body, e.g., count statements
        let stmt_count = node.block.stmts.len();
        self.abstracted
            .push_str(&format!("  // {} statements\n", stmt_count));
        self.abstracted.push_str("}\n\n");
    }

    fn visit_item_struct(&mut self, node: &'ast ItemStruct) {
        self.abstracted
            .push_str(&format!("struct {} {{\n", node.ident));
        for field in &node.fields {
            if let Some(ident) = &field.ident {
                self.abstracted.push_str(&format!("  {}: Type,\n", ident));
            }
        }
        self.abstracted.push_str("}\n\n");
    }

    // Add more visit methods for other items as needed
}
