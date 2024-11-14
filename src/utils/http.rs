use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use anyhow::Result;
use tracing::info;
use tokio::fs;

#[derive(Debug, Deserialize, Serialize)]
pub struct HttpResponse<T> {
    pub code: u16,
    pub message: String,
    pub body: T,
}

impl<T> HttpResponse<T> {
    pub fn new(code: u16, message: String, body: T) -> Self {
        Self { code, message, body }
    }
}


pub async fn download_audio(url: &str, dest_dir: &PathBuf) -> Result<PathBuf> {
    info!("Starting download from URL: {}", url);
    
    // Extract filename from URL
    let filename = url.split('/').last()
        .ok_or_else(|| anyhow::anyhow!("Invalid URL: no filename found"))?;
    
    let dest_path = dest_dir.join(filename);
    info!("Destination path: {:?}", dest_path);

    // Create destination directory (if it doesn't exist)
    if !dest_dir.exists() {
        fs::create_dir_all(dest_dir).await
            .map_err(|e| anyhow::anyhow!("Failed to create directory: {}", e))?;
    }

    // Send HTTP GET request
    let response = reqwest::get(url).await
        .map_err(|e| anyhow::anyhow!("HTTP request failed: {}", e))?;

    if !response.status().is_success() {
        return Err(anyhow::anyhow!(
            "HTTP request failed with status: {}", 
            response.status()
        ));
    }

    // Read response content
    let bytes = response.bytes().await
        .map_err(|e| anyhow::anyhow!("Failed to read response: {}", e))?;

    // 写入文件
    fs::write(&dest_path, bytes).await
        .map_err(|e| anyhow::anyhow!("Failed to write file: {}", e))?;

    info!("Download completed successfully");
    Ok(dest_path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_download_audio() {
        let url = "https://cdn.myshell.ai/audio/opensource/comparison-with-state-of-the-arts/20240201/comp-cn-1-xtts.mp3";
        let dest = PathBuf::from("./xtts.mp3");
        let result = download_audio(url, &dest).await;
        assert!(result.is_ok());
    }
}
