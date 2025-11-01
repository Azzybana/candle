use walkdir::WalkDir;

pub fn collect_files_with_extension(dir: &str, ext: &str) -> Vec<String> {
    WalkDir::new(dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|entry| entry.path().extension().and_then(|s| s.to_str()) == Some(ext))
        .map(|entry| entry.path().to_string_lossy().to_string())
        .collect()
}

pub fn is_valid_json(content: &str) -> bool {
    serde_json::from_str::<serde_json::Value>(content).is_ok()
}

pub fn is_valid_rust_code(content: &str) -> bool {
    syn::parse_str::<syn::File>(content).is_ok()
}

pub fn filter_files_by_content<P>(files: &[String], predicate: P) -> Vec<String>
where
    P: Fn(&str) -> bool,
{
    files.iter()
        .filter(|file| {
            if let Ok(content) = std::fs::read_to_string(file) {
                predicate(&content)
            } else {
                false
            }
        })
        .cloned()
        .collect()
}