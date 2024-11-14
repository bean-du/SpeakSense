use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use std::fmt::Display;
use crate::web::handlers::asr::PathType;


#[derive(Debug, Clone, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum TaskType {
    Transcribe,
    VoiceprintRecognition,
    NoiseReduction,
    // more task types can be added in the future
}

impl Display for TaskType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TaskPriority {
    Critical,
    High,
    Normal,
    Low,
}

impl Default for TaskPriority {
    fn default() -> Self {
        Self::Normal
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskConfig {
    pub task_type: TaskType,
    pub input_path: String,
    pub path_type: PathType,
    pub callback_type: CallbackType,
    pub params: TaskParams,
    pub priority: TaskPriority,
    pub retry_count: u32,
    pub max_retries: u32,
    pub timeout: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "params")]
pub enum TaskParams {
    Transcribe(TranscribeParams),
    VoiceprintRecognition(VoiceprintParams),
    NoiseReduction(NoiseReductionParams),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscribeParams {
    pub language: Option<String>,
    pub speaker_diarization: bool,
    pub emotion_recognition: bool,
    pub filter_dirty_words: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceprintParams {
    // future implementation
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoiseReductionParams {
    // future implementation
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Task {
    pub id: String,
    pub status: TaskStatus,
    pub config: TaskConfig,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub started_at: Option<DateTime<Utc>>,
    pub completed_at: Option<DateTime<Utc>>,
    pub result: Option<TaskResult>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TaskStatus {
    Pending,
    Processing,
    Completed,
    Failed(String),
    Retrying,
    TimedOut,
}

impl TryFrom<String> for TaskStatus {
    type Error = String;
    fn try_from(status: String) -> Result<Self, Self::Error> {
        match status.as_str() {
            "Pending" => Ok(TaskStatus::Pending),
            "Processing" => Ok(TaskStatus::Processing),
            "Completed" => Ok(TaskStatus::Completed),
            "Failed" => Ok(TaskStatus::Failed(String::new())),
            "Retrying" => Ok(TaskStatus::Retrying),
            "TimedOut" => Ok(TaskStatus::TimedOut),
            _ => Err(format!("Invalid task status: {}", status)),
        }
    }
}

impl Display for TaskStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "result")]
pub enum TaskResult {
    Transcribe(TranscribeResult),
    VoiceprintRecognition(VoiceprintResult),
    NoiseReduction(NoiseReductionResult),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscribeResult {
    pub text: String,
    pub segments: Vec<TranscribeSegment>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscribeSegment {
    pub text: String,
    pub speaker_id: Option<usize>,
    pub start_time: f64,
    pub end_time: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceprintResult {
    // future implementation
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoiseReductionResult {
    // future implementation
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "config")]
pub enum CallbackType {
    Http { url: String },
    Function { name: String },
    Event,
    None,
} 