use crate::training::code_trainer::simd::simd_process_tokens;
use anyhow::Result;
use tokenizers::Tokenizer;
use trash_parallelism::parallel::parallel_map;

pub struct RustTokenizer {
    tokenizer: Tokenizer,
}

impl RustTokenizer {
    pub fn new(model_path: &str) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(model_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
        Ok(Self { tokenizer })
    }

    pub fn tokenize_batch(&self, batch: &[String]) -> Result<Vec<Vec<i32>>> {
        // Parallel tokenization for speed
        let results: Vec<Result<Vec<i32>>> = parallel_map(batch.to_vec(), |section| {
            let encoding = self
                .tokenizer
                .encode(section.as_str(), false)
                .map_err(|e| anyhow::anyhow!("Encoding failed: {}", e))?;
            let ids: Vec<i32> = encoding.get_ids().iter().map(|&x| x as i32).collect();
            Ok(ids)
        });

        // Collect and apply SIMD processing if needed
        let mut tokenized = Vec::new();
        for result in results {
            let mut ids = result?;
            simd_process_tokens(&mut ids); // Custom SIMD optimization
            tokenized.push(ids);
        }

        Ok(tokenized)
    }
}
