use anyhow::Result;
use trash_parallelism::chars::processing::read_file_to_string_async;

pub async fn read_markdown(path: &str) -> Result<String> {
    let content = read_file_to_string_async(path).await?;
    Ok(content)
}
