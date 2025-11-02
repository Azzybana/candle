#![allow(dead_code)]

use trash_parallelism::sys::path::find_files_parallel;
use trash_parallelism::sys::Timer;
use trash_parallelism::chars::core::deduplicate_lines;
use trash_parallelism::parallel::parallel_filter;

pub fn collect_files_with_extension(dir: &str, ext: &str) -> Vec<String> {
    let _timer = Timer::new("collect_files_with_extension");

    // Use parallel file finding for better performance
    let pattern = format!("*.{}", ext);
    find_files_parallel(dir, &pattern).unwrap_or_default()
}

pub fn is_valid_json(content: &str) -> bool {
    serde_json::from_str::<serde_json::Value>(content).is_ok()
}

pub fn is_valid_rust_code(content: &str) -> bool {
    syn::parse_str::<syn::File>(content).is_ok()
}

pub fn filter_files_by_content<P>(files: &[String], predicate: P) -> Vec<String>
where
    P: Fn(&str) -> bool + Send + Sync,
{
    parallel_filter(files.to_vec(), |file| {
        std::fs::read_to_string(file)
            .map(|content| predicate(&content))
            .unwrap_or(false)
    })
}

pub fn deduplicate_text_samples(samples: Vec<String>) -> Vec<String> {
    let _timer = Timer::new("deduplicate_text_samples");

    // Join all samples with newlines, deduplicate lines, then split back
    let combined = samples.join("\n");
    let deduplicated = deduplicate_lines(&combined);
    deduplicated.lines().map(|s| s.to_string()).collect()
}