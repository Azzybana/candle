## eval

```bash
cargo run --example llava -- --image-file "llava_logo.png" --prompt "is this a cat?" --hf # default args, use  llava-hf/llava-v1.6-vicuna-7b-hf. image-file is required^_^
cargo run --example llava -- --model-path liuhaotian/llava-v1.6-vicuna-7b  --image-file "llava_logo.png" --prompt "is this a cat?" # use liuhaotian/llava-v1.6-vicuna-7b, tokenizer setup should be done
```