# Training Module Documentation

This directory contains the training pipeline components for Chloe, a Rust-based AI model trainer. Each module handles a specific aspect of data preparation, processing, and model training.

## Modules Overview

### `code_trainer.rs`
**Purpose**: Prepares training data from Rust source code files.

**Key Features**:
- Collects `.rs` files from a project directory
- Validates Rust code syntax using `syn`
- Abstracts code structure (functions, structs) into simplified representations
- Processes files in parallel using `trash_parallelism::parallel_map_async`
- Generates tokenized training samples for code understanding tasks

**Usage**: Used to create datasets for code generation, completion, or analysis models.

### `combinator.rs`
**Purpose**: Combines multiple SafeTensors files for dataset merging.

**Key Features**:
- Concatenates tensors from multiple SafeTensors files
- Handles both shared and unique tensor names across files
- Sequential combination for memory efficiency
- Supports batch processing of multiple datasets

**Usage**: Useful for merging small training datasets into larger ones for better training efficiency.

### `common.rs`
**Purpose**: Shared utilities and helper functions used across training modules.

**Key Features**:
- `tokenize_texts()`: Parallel tokenization using `trash_parallelism::parallel_map`
- `load_tokenizer()`: Loads tokenizer from JSON file
- `create_training_dir()`: Creates output directories for training data
- `save_training_data()`: Async saving of tensors and metadata

**Usage**: Core utilities imported by all training modules.

### `conversation_trainer.rs`
**Purpose**: Prepares training data from conversational JSON files.

**Key Features**:
- Processes JSON files containing user-assistant conversation pairs
- Parallel file reading and parsing with `parallel_map_async`
- Formats conversations with special tokens (`<|user|>`, `<|assistant|>`)
- Deduplicates conversation samples

**Usage**: Creates datasets for chatbot or conversational AI training.

### `conversion.rs`
**Purpose**: Handles model format conversions between different serialization formats.

**Key Features**:
- `convert_safetensors_to_gguf()`: Converts SafeTensors to GGUF with quantization
- `convert_safetensors_to_onnx()`: Exports to ONNX format
- `convert_gguf_to_onnx()`: Converts GGUF models to ONNX
- Supports metadata preservation and tensor quantization

**Usage**: Enables model deployment across different inference engines.

### `filters.rs`
**Purpose**: File discovery, validation, and filtering utilities.

**Key Features**:
- `collect_files_with_extension()`: Parallel file discovery using `find_files_parallel`
- `filter_files_by_content()`: Parallel content validation with `parallel_filter`
- `is_valid_json()` / `is_valid_rust_code()`: Content validation functions
- `deduplicate_text_samples()`: Removes duplicate text samples

**Usage**: Used by all trainer modules for efficient file processing and data cleaning.

### `reasoning_trainer.rs`
**Purpose**: Prepares training data from reasoning problem JSON files.

**Key Features**:
- Processes JSON files containing problem-solution pairs
- Parallel processing of reasoning datasets
- Formats problems with "Problem:" and "Solution:" prefixes
- Handles mathematical or logical reasoning tasks

**Usage**: Creates datasets for reasoning, math, or analytical AI models.

### `text_trainer.rs`
**Purpose**: Handles general text data preparation and basic model training.

**Key Features**:
- Reads and processes `.txt` files in parallel
- Includes a simple text classification training loop using Candle
- Embedding + Linear layer architecture
- CPU-based training with AdamW optimizer

**Usage**: General-purpose text processing and basic model training demonstrations.

## Architecture Notes

- **Parallel Processing**: All I/O operations use `trash_parallelism` for maximum performance
- **Async Design**: File operations are async to prevent blocking
- **Memory Efficiency**: Streaming processing and deduplication reduce memory usage
- **Modularity**: Each trainer focuses on one data type for easy extension
- **Error Handling**: Robust error handling with `anyhow` for clean error propagation

## Dependencies

- `trash_parallelism`: High-performance parallel processing utilities
- `candle`: Machine learning framework for Rust
- `tokenizers`: Text tokenization
- `safetensors`: Model serialization
- `serde`: JSON serialization/deserialization
- `syn`: Rust code parsing for code abstraction

## Performance Characteristics

- **Scalability**: Linear scaling with CPU cores for parallel operations
- **I/O Bound**: Optimized for high-throughput file processing
- **Memory Usage**: Efficient streaming and deduplication
- **CPU Utilization**: Maximizes available cores for compute-intensive tasks