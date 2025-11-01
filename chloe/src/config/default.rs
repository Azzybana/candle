use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize, Debug)]
pub struct ChloeConfig {
    pub chloe: Config,
    pub training: Option<TrainingConfig>,
    pub chat: Option<ChatConfig>,
    pub code: Option<CodeConfig>,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct Config {
    /// Path to the model file (e.g., .gguf, .safetensors)
    pub model: String,
    /// Path to the tokenizer file (e.g., .json)
    pub tokenizer: String,
    /// Path to the prompt file (e.g., .md, .txt)
    pub prompt: String,
    /// The length of the sample to generate (in tokens).
    pub sample_len: usize,
    /// The temperature used to generate samples, use 0 for greedy sampling.
    pub temperature: f64,
    /// Nucleus sampling probability cutoff.
    pub top_p: Option<f64>,
    /// Only sample among the top K samples.
    pub top_k: Option<usize>,
    /// The seed to use when generating random samples.
    pub seed: u64,
    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    pub repeat_penalty: f32,
    /// The context size to consider for the repeat penalty.
    pub repeat_last_n: usize,
    /// Maximum context length in tokens.
    pub max_context_length: usize,
    /// Template for formatting the prompt.
    pub prompt_template: String,
    /// List of end-of-sequence tokens.
    pub eos_tokens: Vec<String>,
}

#[derive(Deserialize, Serialize, Debug)]
pub struct TrainingConfig {
    /// Path to the source SafeTensors file
    pub source_safetensors: String,
    /// Path to the output GGUF file
    pub output_gguf: String,
    /// Optional path to metadata JSON file
    pub metadata: Option<String>,
}

#[derive(Deserialize, Serialize, Debug)]
pub struct ChatConfig {
    /// Custom prompt for chat mode
    pub prompt: String,
}

#[derive(Deserialize, Serialize, Debug)]
pub struct CodeConfig {
    /// Custom prompt for code mode
    pub prompt: String,
}

impl ChloeConfig {
    pub fn default_config() -> Self {
        ChloeConfig {
            chloe: Config {
                model: "Qwen3-4B-Function-Calling.Pro.gguf".to_string(),
                tokenizer: "tokenizer.json".to_string(),
                prompt: "prompt.md".to_string(),
                sample_len: 1000,
                temperature: 0.7,
                top_p: Some(0.8),
                top_k: Some(20),
                seed: 299792458,
                repeat_penalty: 1.1,
                repeat_last_n: 64,
                max_context_length: 262144,
                prompt_template: "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n".to_string(),
                eos_tokens: vec!["<|im_end|>".to_string(), "<|endoftext|>".to_string()],
            },
            training: Some(TrainingConfig {
                source_safetensors: "model.safetensors".to_string(),
                output_gguf: "model.gguf".to_string(),
                metadata: Some("metadata.json".to_string()),
            }),
            chat: Some(ChatConfig {
                prompt: "You are a helpful AI assistant. Respond to the user's message.".to_string(),
            }),
            code: Some(CodeConfig {
                prompt: "You are a code generation assistant. Generate code based on the user's request.".to_string(),
            }),
        }
    }
}