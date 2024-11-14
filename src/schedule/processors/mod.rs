pub mod transcribe;

use async_trait::async_trait;
use anyhow::Result;
use crate::schedule::types::{Task, TaskResult, TaskType, TaskParams};

pub use transcribe::TranscribeProcessor;

#[async_trait]
pub trait TaskProcessor: Send + Sync {
    fn task_type(&self) -> TaskType;
    async fn process(&self, task: &Task) -> Result<TaskResult>;
    fn validate_params(&self, params: &TaskParams) -> Result<()>;
    async fn cancel(&self, task: &Task) -> Result<()>;
    async fn cleanup(&self, task: &Task) -> Result<()>;
} 