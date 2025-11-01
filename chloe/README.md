# Chloe

A standalone Rust project for running quantized Qwen3 models locally.

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

This creates `data/config.toml` with default values and `data/prompt.md` if it doesn't exist.

To generate just the prompt file:

```bash
cargo run -- --generate-prompt
```

This creates `data/prompt.md`.

## Configuration

The `config.toml` file has the following structure:

```toml
[chloe]
model = "model.gguf"  # Path to model file (.gguf, .safetensors, etc.)
tokenizer = "tokenizer.json"  # Path to tokenizer file (.json)
prompt = "prompt.md"  # Path to prompt file (.md, .txt, etc.)
```

Paths are relative to the config file's directory.

## Options

- `--model`: Override model path
- `--tokenizer`: Override tokenizer path
- `--prompt`: Override prompt file path
- `--sample-len`: Length of the sample to generate (default: 1000)
- `--temperature`: Sampling temperature (default: 0.8)
- `--top-p`: Nucleus sampling probability cutoff
- `--top-k`: Only sample among the top K samples
- `--seed`: Random seed (default: 299792458)
- `--tracing`: Enable tracing
- `--split-prompt`: Process prompt elements separately
- `--cpu`: Run on CPU
- `--repeat-penalty`: Penalty for repeating tokens (default: 1.1)
- `--repeat-last-n`: Context size for repeat penalty (default: 64)
- `--which`: Model size (default: 4b)
- `--generate-config`: Generate default config.toml (destructive: overwrites existing)
- `--generate-prompt`: Generate default prompt.md (destructive: overwrites existing)
- `--use-config`: Specify specific config.toml file to use
- `--use-prompt`: Specify specific prompt file to use
- `--use-model`: Specify specific model file to use
- `--use-tokenizer`: Specify specific tokenizer file to use

Precedence: Command line options override config file values.

If `--use-*` options are specified, the files must exist; no defaults are created.