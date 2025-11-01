# Chloe

A standalone Rust project for running quantized Qwen3 models locally with multiple modes.

## Usage

The project searches for `config.toml` in the following order:
- Current directory
- `data/`
- `.config/`
- `config/`
- `model/`

If a valid config is found, it loads it. Otherwise, prints help and exits.

You can generate a default config with:

```bash
cargo run -- --generate-config
```

This creates `data/config.toml` with default values and prompt files if they don't exist.

## Modes

Chloe supports different modes for specialized tasks:

- **Default**: General inference with custom prompts.
- **Chat**: Conversational AI with chat-specific prompts.
- **Code**: Code generation and assistance.
- **Training**: Convert SafeTensors models to GGUF format.

Use flags like `--chat`, `--code`, `--training` to select the mode.

## Configuration

The `config.toml` file has the following structure:

```toml
[chloe]
model = "Qwen3-4B-Function-Calling.Pro.gguf"  # Path to model file (.gguf, .safetensors, etc.)
tokenizer = "tokenizer.json"  # Path to tokenizer file (.json)
prompt = "prompt.md"  # Path to prompt file (.md, .txt, etc.)
sample_len = 1000  # Length of the sample to generate (in tokens)
temperature = 0.7  # Sampling temperature (0 for greedy)
top_p = 0.8  # Nucleus sampling probability cutoff
top_k = 20  # Only sample among the top K samples
seed = 299792458  # Random seed
repeat_penalty = 1.1  # Penalty for repeating tokens
repeat_last_n = 64  # Context size for repeat penalty
max_context_length = 262144  # Maximum context length in tokens
prompt_template = "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"  # Template for prompt formatting
eos_tokens = ["<|im_end|>", "<|endoftext|>"]  # List of end-of-sequence tokens

[training]
source_safetensors = "model.safetensors"  # Source SafeTensors file
output_gguf = "model.gguf"  # Output GGUF file
metadata = "metadata.json"  # Optional metadata JSON

[chat]
prompt = "You are a helpful AI assistant. Respond to the user's message."  # Chat mode prompt

[code]
prompt = "You are a code generation assistant. Generate code based on the user's request."  # Code mode prompt
```

Paths are relative to the config file's directory.

## Training Conversion

To convert a SafeTensors model to GGUF:

```bash
cargo run -- --training
```

This uses the paths specified in the `[training]` section of the config.

## Options

- `--model`: Override model path
- `--tokenizer`: Override tokenizer path
- `--prompt`: Override prompt file path
- `--sample-len`: Length of the sample to generate (in tokens)
- `--temperature`: Sampling temperature (default: 0.7)
- `--top-p`: Nucleus sampling probability cutoff
- `--top-k`: Only sample among the top K samples
- `--seed`: Random seed (default: 299792458)
- `--tracing`: Enable tracing
- `--split-prompt`: Process prompt elements separately
- `--cpu`: Run on CPU
- `--repeat-penalty`: Penalty for repeating tokens (default: 1.1)
- `--repeat-last-n`: Context size for repeat penalty (default: 64)
- `--generate-config`: Generate default config.toml (destructive: overwrites existing)
- `--generate-prompt`: Generate default prompt.md (destructive: overwrites existing)
- `--use-config`: Specify specific config.toml file to use
- `--use-prompt`: Specify specific prompt file to use
- `--use-model`: Specify specific model file to use
- `--use-tokenizer`: Specify specific tokenizer file to use
- `--chat`: Run in chat mode
- `--code`: Run in code mode
- `--training`: Run training conversion
- `--translate`: Run in translate mode

Precedence: Command line options override config file values.

If `--use-*` options are specified, the files must exist; no defaults are created.