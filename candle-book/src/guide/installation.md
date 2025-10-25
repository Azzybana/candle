# Installation

## 1. Create a new rust app or library

```bash
cargo new myapp
cd myapp
```

## 2. Add the correct candle version

### Standard

```bash
cargo add --git https://github.com/huggingface/candle.git candle-core
```

### MKL

You can also see the `mkl` feature which can get faster inference on CPU.

Add the `candle-core` crate with the mkl feature:

```bash
cargo add --git https://github.com/huggingface/candle.git candle-core --features "mkl"
```

### Metal

Metal is exclusive to MacOS.

Add the `candle-core` crate with the metal feature:

```bash
cargo add --git https://github.com/huggingface/candle.git candle-core --features "metal"
```

## 3. Building

Run `cargo build` to make sure everything can be correctly built.

```bash
cargo build
```
