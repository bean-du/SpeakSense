use async_trait::async_trait;
use anyhow::Result;
use std::sync::Arc;
use tokio::sync::mpsc;
use crate::web::handlers::asr::PathType;
use tracing::{info, warn, error};

use crate::asr::{whisper::WhisperAsr, AsrParams};
use crate::schedule::types::{
    Task, TaskType, TaskResult, TaskParams, TranscribeParams,
    TranscribeResult, TranscribeSegment
};
use crate::utils::http::download_audio;
use super::TaskProcessor;
use std::path::PathBuf;
use std::fs;
use crate::AUDIO_PATH;
use crate::audio::DenoiseConfig;
use crate::audio::parse_audio_file_stream;

#[derive(Clone)]
pub struct TranscribeProcessor {
    asr: Arc<WhisperAsr>,
}

impl TranscribeProcessor {
    pub fn new(asr: Arc<WhisperAsr>) -> Self {
        Self { asr }
    }

    pub async fn process_audio(&self, task: &Task, params: &TranscribeParams) -> Result<TranscribeResult> {
        let path_type = task.config.path_type.clone();
        let dest = match path_type {
            PathType::Url => {
                // Ensure the download path is exists
                let download_dir = PathBuf::from(AUDIO_PATH.as_str());
                if let Err(e) = fs::create_dir_all(&download_dir) {
                    error!("Failed to create download directory: {}", e);
                    return Err(anyhow::anyhow!("Failed to create download directory: {}", e));
                }

                // Download the Audio file
                info!("Attempting to download audio from: {}", task.config.input_path);
                let dest = match download_audio(&task.config.input_path, &download_dir).await {
                    Ok(dest) => {
                        info!("Successfully downloaded audio to: {:?}", dest);
                        dest
                    },
                    Err(e) => {
                        error!("Failed to download audio from {}: {}", task.config.input_path, e);
                        return Err(anyhow::anyhow!("Failed to download audio from {}: {}", task.config.input_path, e));
                    }
                };
                dest
            }
            PathType::Local => {
                PathBuf::from(&task.config.input_path)
            }
        };
        
       
        let (tx, mut rx) = mpsc::channel(32);
        let (audio_tx, mut audio_rx) = mpsc::channel(32);
        let config = DenoiseConfig::default();
        
        let mut asr_params = AsrParams::new();
        asr_params.set_language(params.language.clone());
        asr_params.set_speaker_diarization(params.speaker_diarization);
        asr_params.set_stream_mode(true);
        
        let asr = self.asr.clone();

        
        // Start audio processor
        let dest_clone = dest.clone();
        let audio_tx = Arc::new(tokio::sync::Mutex::new(audio_tx));
        
        tokio::spawn(async move {
            info!("Starting audio processing for file: {}", dest_clone.display());
            
            let audio_tx = audio_tx.clone();
            // Read audio file stream
            match parse_audio_file_stream(
                &dest_clone,
                &config,
                move |chunk| {
                    let audio_tx = audio_tx.clone();
                    tokio::spawn(async move {
                        if let Err(e) = audio_tx.lock().await.send(chunk).await {
                            error!("Failed to send audio chunk: {}", e);
                        }
                    });
                },
            ).await {
                Ok(_) => info!("Audio processing completed for: {}", dest_clone.display()),
                Err(e) => error!("Failed to process audio: {}", e),
            }
        });

        let state = asr.create_state()?;
        // Start Transcribe task
        tokio::spawn(async move {
            info!("Starting transcription task");
            let mut buffer = Vec::new();
            const BUFFER_SIZE: usize = 16000 * 30; 

            while let Some(chunk) = audio_rx.recv().await {
                buffer.extend(chunk);
                
                if buffer.len() >= BUFFER_SIZE {
                    let state_clone = state.clone();
                    match asr.transcribe_with_state(state_clone, buffer.clone(), asr_params.clone()).await {
                        Ok(result) => {
                            info!("Transcribed segment: {}", result.full_text);
                            if let Err(e) = tx.send(result).await {
                                error!("Failed to send transcribe result: {}", e);
                                break;
                            }
                        },
                        Err(e) => {
                            error!("Failed to transcribe audio chunk: {}", e);
                        }
                    }
                    buffer.clear();
                }
            }

            if !buffer.is_empty() {
                let state_clone = state.clone();
                match asr.transcribe_with_state(state_clone, buffer.clone(), asr_params.clone()).await {
                    Ok(result) => {
                        info!("Transcribed final segment: {}", result.full_text);
                        if let Err(e) = tx.send(result).await {
                            error!("Failed to send final transcribe result: {}", e);
                        }
                    },
                    Err(e) => {
                        error!("Failed to transcribe final audio chunk: {}", e);
                    }
                }
            }
        });

        // Collect result
        let mut full_text = String::new();
        let mut all_segments = Vec::new();
        
        while let Some(result) = rx.recv().await {
            full_text.push_str(&result.full_text);
            all_segments.extend(result.segments);
        }

        // Clean audio file
        if let Err(e) = fs::remove_file(&dest) {
            warn!("Failed to remove temporary file: {}", e);
        }

        Ok(TranscribeResult {
            text: full_text,
            segments: all_segments.into_iter().map(|s| TranscribeSegment {
                text: s.text,
                speaker_id: Some(s.speaker_id),
                start_time: s.start,
                end_time: s.end,
            }).collect(),
        })
    }
}

#[async_trait]
impl TaskProcessor for TranscribeProcessor {
    fn task_type(&self) -> TaskType {
        TaskType::Transcribe
    }

    async fn process(&self, task: &Task) -> Result<TaskResult> {
        let params = match &task.config.params {
            TaskParams::Transcribe(p) => p,
            _ => return Err(anyhow::anyhow!("Invalid task params")),
        };

        info!("Processing transcribe task {} with params: {:?}", task.id, params);

        match self.process_audio(task, params).await {
            Ok(result) => {
                info!("Successfully processed task {}", task.id);
                Ok(TaskResult::Transcribe(result))
            }
            Err(e) => {
                warn!("Failed to process task {}: {}", task.id, e);
                Err(e)
            }
        }
    }

    fn validate_params(&self, params: &TaskParams) -> Result<()> {
        match params {
            TaskParams::Transcribe(p) => {
                // validate language parameter
                if let Some(lang) = &p.language {
                    if !["zh", "en", "ja"].contains(&lang.as_str()) {
                        return Err(anyhow::anyhow!("Unsupported language: {}", lang));
                    }
                }

                // validate input file - get from TaskConfig
                if let TaskParams::Transcribe(_) = params {
                    // note: validation should be done when creating task, because we cannot access TaskConfig here
                    // we only validate language parameter here
                    Ok(())
                } else {
                    Err(anyhow::anyhow!("Invalid task params type"))
                }
            }
            _ => Err(anyhow::anyhow!("Invalid task params type")),
        }
    }

    async fn cancel(&self, task: &Task) -> Result<()> {
        // ASR does not support canceling ongoing tasks
        warn!("Cancel operation is not supported for task {}", task.id);
        Ok(())
    }

    async fn cleanup(&self, task: &Task) -> Result<()> {
        // clean up temporary file
        if PathBuf::from(&task.config.input_path).exists() {
            info!("Cleaning up temporary file: {}", task.config.input_path);
            if let Err(e) = std::fs::remove_file(&task.config.input_path) {
                warn!("Failed to remove temporary file: {}", e);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schedule::types::TaskConfig;
    use std::path::PathBuf;
    use crate::schedule::types::{CallbackType, TaskParams, TaskPriority, TaskStatus};
    use chrono::Utc;
    use crate::schedule::types::TranscribeParams;
    use crate::asr::whisper::WhisperAsr;
    use crate::web::handlers::asr::PathType;

    #[tokio::test]
    async fn test_transcribe_processor() -> Result<()> {
        let test_file = PathBuf::from("./test/1.wav");

        // create processor
        let asr = Arc::new(WhisperAsr::new("./models/ggml-large-v3.bin".to_string())?);
        let processor = TranscribeProcessor::new(asr);

        // create test task
        let task = Task {
            id: "test-task".to_string(),
            status: TaskStatus::Pending,
            config: TaskConfig {
                task_type: TaskType::Transcribe,
                input_path: test_file.to_str().unwrap().to_string(),
                path_type: PathType::Url,
                callback_type: CallbackType::Http { url: "http://localhost:8000/callback".to_string() },
                params: TaskParams::Transcribe(TranscribeParams {
                    language: Some("zh".to_string()),
                    speaker_diarization: true,
                    emotion_recognition: false,
                    filter_dirty_words: false,
                }),
                priority: TaskPriority::Normal,
                retry_count: 0,
                max_retries: 3,
                timeout: Some(300),
            },
            created_at: Utc::now(),
            updated_at: Utc::now(),
            started_at: None,
            completed_at: None,
            result: None,
            error: None,
        };

        // validate params
        processor.validate_params(&task.config.params)?;

        // process task
        let result = processor.process(&task).await?;

        // validate result
        match result {
            TaskResult::Transcribe(result) => {
                assert!(!result.text.is_empty());
                assert!(!result.segments.is_empty());
            }
            _ => panic!("Unexpected result type"),
        }

        // clean up test file
        processor.cleanup(&task).await?;
        assert!(!test_file.exists());

        Ok(())
    }
} 