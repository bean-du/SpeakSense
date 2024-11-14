use anyhow::Result;
use serde::{Serialize, Deserialize};
use async_trait::async_trait;
use whisper_rs::WhisperState;
use std::sync::{Arc, Mutex};

pub mod whisper;    

#[derive(Debug, Clone)]
pub struct AsrParams {
    pub language: Option<String>,
    pub speaker_diarization: bool,
    pub stream_mode: bool,           
    pub min_segment_length: usize,   
}

impl AsrParams {
    pub fn new() -> Self {
        Self {
            language: None,
            speaker_diarization: false,
            stream_mode: false,
            min_segment_length: 10,
        }
    }

    pub fn set_language(&mut self, language: Option<String>) {
        self.language = language;
    }

    pub fn set_speaker_diarization(&mut self, enable: bool) {
        self.speaker_diarization = enable;
    }

    pub fn set_stream_mode(&mut self, enable: bool) {
        self.stream_mode = enable;
    }

    pub fn set_min_segment_length(&mut self, length: usize) {
        self.min_segment_length = length;
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscribeSegment {
    pub text: String,
    pub speaker_id: usize,    
    pub start: f64,    
    pub end: f64,      
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TranscribeResult {
    pub segments: Vec<TranscribeSegment>,
    pub full_text: String,
}

#[async_trait]
pub trait AsrEngine: Send + Sync {
    fn create_state(&self) -> Result<Arc<Mutex<Box<WhisperState<'static>>>>>;

    async fn transcribe_with_state(
        &self, 
        state: Arc<Mutex<Box<WhisperState<'static>>>>,
        audio: Vec<f32>, 
        params: AsrParams
    ) -> Result<TranscribeResult>;

    async fn transcribe(&self, audio: Vec<f32>, params: AsrParams) -> Result<TranscribeResult> {
        let state = self.create_state()?;
        self.transcribe_with_state(state, audio, params).await
    }
}