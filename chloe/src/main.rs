mod cli;
mod config;
mod markdown;
mod token_output_stream;
mod training;
mod mode;

use std::io::Write;

use candle::Tensor;
use candle::quantized::gguf_file;
use candle_transformers::generation::{LogitsProcessor, Sampling};

use candle_transformers::models::quantized_qwen3::ModelWeights as Qwen3;
use token_output_stream::TokenOutputStream;

use crate::config::default::ChloeConfig;
use clap::{CommandFactory, Parser};
use cli::Args;

fn device(_cpu: bool) -> candle::Result<candle::Device> {
    Ok(candle::Device::Cpu)
}

fn format_size(size_in_bytes: usize) -> String {
    if size_in_bytes < 1_000 {
        format!("{size_in_bytes}B")
    } else if size_in_bytes < 1_000_000 {
        format!("{:.2}KB", size_in_bytes as f64 / 1e3)
    } else if size_in_bytes < 1_000_000_000 {
        format!("{:.2}MB", size_in_bytes as f64 / 1e6)
    } else {
        format!("{:.2}GB", size_in_bytes as f64 / 1e9)
    }
}

fn main() -> anyhow::Result<()> {
    smol::block_on(async_main())
}

async fn async_main() -> anyhow::Result<()> {
    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::prelude::*;

    let args = Args::parse();

    // Handle generate commands
    if args.generate_config {
        std::fs::create_dir_all("data")?;
        ChloeConfig::generate_config_file("data/config.toml")?;
        println!("Generated data/config.toml");
        return Ok(());
    }

    if args.generate_prompt {
        std::fs::create_dir_all("data")?;
        ChloeConfig::generate_prompt_file("data/prompt.md")?;
        println!("Generated data/prompt.md");
        return Ok(());
    }

    // Determine config path
    let config_path = if let Some(path) = &args.use_config {
        path.clone()
    } else {
        ChloeConfig::find_config().await.unwrap_or_else(|| {
            Args::command().print_help().unwrap();
            std::process::exit(1);
        })
    };

    // Load config
    let config = ChloeConfig::load_from_file(&config_path).await?;
    let config_dir = std::path::Path::new(&config_path).parent();

    let _guard = if args.tracing {
        let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
        tracing_subscriber::registry().with(chrome_layer).init();
        Some(guard)
    } else {
        None
    };

    println!(
        "avx: {}, neon: {}, simd128: {}, f16c: {}",
        candle::utils::with_avx(),
        candle::utils::with_neon(),
        candle::utils::with_simd128(),
        candle::utils::with_f16c()
    );
    println!(
        "temp: {:.2} repeat-penalty: {:.2} repeat-last-n: {}",
        args.effective_temperature(&config.chloe),
        args.effective_repeat_penalty(&config.chloe),
        args.effective_repeat_last_n(&config.chloe)
    );

    // Dispatch to modes
    if args.training {
        training::run_training(&config).await?;
        return Ok(());
    }

    if args.chat {
        mode::run_mode(&config, mode::Mode::Chat).await?;
        return Ok(());
    }

    if args.code {
        mode::run_mode(&config, mode::Mode::Code).await?;
        return Ok(());
    }

    if args.translate {
        println!("Translate mode not implemented yet.");
        return Ok(());
    }

    let model_path = args.model_path(&config.chloe, config_dir);
    let mut file = std::fs::File::open(&model_path)?;
    let start = std::time::Instant::now();
    let device = device(args.cpu)?;

    let mut model = {
        let model =
            gguf_file::Content::read(&mut file).map_err(|e| e.with_path(model_path.clone()))?;
        let mut total_size_in_bytes = 0;
        for (_, tensor) in model.tensor_infos.iter() {
            let elem_count = tensor.shape.elem_count();
            total_size_in_bytes +=
                elem_count * tensor.ggml_dtype.type_size() / tensor.ggml_dtype.block_size();
        }
        println!(
            "loaded {:?} tensors ({}) in {:.2}s",
            model.tensor_infos.len(),
            &format_size(total_size_in_bytes),
            start.elapsed().as_secs_f32(),
        );
        Qwen3::from_gguf(model, &mut file, &device)?
    };
    println!("model built");

    let tokenizer = args.tokenizer(&config.chloe, config_dir)?;
    let tokenizer_path = args.tokenizer_path(&config.chloe, config_dir);
    let vocab = crate::config::read_config::load_vocab(&tokenizer_path)?;
    let mut tos = TokenOutputStream::new(tokenizer, vocab);
    let prompt_str = args.prompt_content(&config.chloe, config_dir).await?;

    let prompt_str = config
        .chloe
        .prompt_template
        .replace("{prompt}", &prompt_str);
    print!("formatted prompt: {}", &prompt_str);

    let tokens = tos
        .tokenizer()
        .encode(prompt_str, true)
        .map_err(anyhow::Error::msg)?;

    let tokens = tokens.get_ids();

    let to_sample = args.effective_sample_len(&config.chloe).saturating_sub(1);

    let mut all_tokens = vec![];

    let mut logits_processor = {
        let temperature = args.effective_temperature(&config.chloe);
        let sampling = if temperature <= 0. {
            Sampling::ArgMax
        } else {
            match (
                args.effective_top_k(&config.chloe),
                args.effective_top_p(&config.chloe),
            ) {
                (None, None) => Sampling::All { temperature },
                (Some(k), None) => Sampling::TopK { k, temperature },
                (None, Some(p)) => Sampling::TopP { p, temperature },
                (Some(k), Some(p)) => Sampling::TopKThenTopP { k, p, temperature },
            }
        };
        LogitsProcessor::from_sampling(args.effective_seed(&config.chloe), sampling)
    };

    let start_prompt_processing = std::time::Instant::now();

    let mut next_token = if !args.split_prompt {
        let input = Tensor::new(tokens, &device)?.unsqueeze(0)?;
        let logits = model.forward(&input, 0)?;
        let logits = logits.squeeze(0)?;
        logits_processor.sample(&logits)?
    } else {
        let mut next_token = 0;
        for (pos, token) in tokens.iter().enumerate() {
            let input = Tensor::new(&[*token], &device)?.unsqueeze(0)?;
            let logits = model.forward(&input, pos)?;
            let logits = logits.squeeze(0)?;
            next_token = logits_processor.sample(&logits)?
        }
        next_token
    };

    let prompt_dt = start_prompt_processing.elapsed();

    all_tokens.push(next_token);

    if let Some(t) = tos.next_token(next_token)? {
        print!("{t}");
        std::io::stdout().flush()?;
    }

    let eos_tokens: Vec<u32> = config
        .chloe
        .eos_tokens
        .iter()
        .filter_map(|t| tos.get_token(t))
        .collect();

    let start_post_prompt = std::time::Instant::now();

    let mut sampled = 0;
    for index in 0..to_sample {
        let input = Tensor::new(&[next_token], &device)?.unsqueeze(0)?;
        let logits = model.forward(&input, tokens.len() + index)?;
        let logits = logits.squeeze(0)?;
        let logits = if args.effective_repeat_penalty(&config.chloe) == 1. {
            logits
        } else {
            let start_at = all_tokens
                .len()
                .saturating_sub(args.effective_repeat_last_n(&config.chloe));
            candle_transformers::utils::apply_repeat_penalty(
                &logits,
                args.effective_repeat_penalty(&config.chloe),
                &all_tokens[start_at..],
            )?
        };
        next_token = logits_processor.sample(&logits)?;
        all_tokens.push(next_token);
        if let Some(t) = tos.next_token(next_token)? {
            print!("{t}");
            std::io::stdout().flush()?;
        }
        sampled += 1;
        if eos_tokens.contains(&next_token) {
            break;
        };
    }

    if let Some(rest) = tos.decode_rest().map_err(candle::Error::msg)? {
        print!("{rest}");
    }

    std::io::stdout().flush()?;
    let dt = start_post_prompt.elapsed();
    println!(
        "\n\n{:4} prompt tokens processed: {:.2} token/s",
        tokens.len(),
        tokens.len() as f64 / prompt_dt.as_secs_f64(),
    );
    println!(
        "{sampled:4} tokens generated: {:.2} token/s",
        sampled as f64 / dt.as_secs_f64(),
    );
    Ok(())
}
