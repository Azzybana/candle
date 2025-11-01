use anyhow::Result;
use std::fs;

pub fn read_markdown(path: &str) -> Result<String> {
    Ok(fs::read_to_string(path)?)
}
