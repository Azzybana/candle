# candle-voxtral: speech recognition

An implementation of Voxtral speech recognition using candle.

## Running the example

Run the example:
```bash
cargo run --example voxtral --features tekken,symphonia,rubato --release
```

## Command line options

- `--cpu`: Run on CPU (default: true, since no GPU support)
- `--input`: Audio file path in wav format. If not provided, a sample file is automatically downloaded from the hub.
- `--model-id`: Model to use (default: `mistralai/Voxtral-Mini-3B-2507`)
