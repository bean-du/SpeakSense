use std::sync::Arc;


pub mod types;
pub mod processors;
pub mod scheduler;
pub mod callback;
// mod tests;

pub use types::{
    Task, TaskType, TaskConfig, TaskParams, TaskStatus, TaskResult,
    TaskPriority, TranscribeParams, TranscribeResult, CallbackType,
};

pub use crate::storage::task::TaskStorage;

pub use processors::TaskProcessor;
pub use processors::transcribe::TranscribeProcessor;

pub use scheduler::{TaskManager, TaskScheduler};

pub async fn create_scheduler(
    storage: impl TaskStorage + Send + Sync + 'static,
    processors: Vec<Box<dyn TaskProcessor>>,
) -> anyhow::Result<TaskScheduler> {
    let mut task_manager = TaskManager::new(Arc::new(storage));
    
    for processor in processors {
        task_manager.register_processor(processor);
    }
    
    Ok(TaskScheduler::new(Arc::new(task_manager)))
}