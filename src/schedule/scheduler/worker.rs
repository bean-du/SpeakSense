use std::sync::Arc;
use tokio::time::{sleep, Duration};
use tracing::{info, error};
use anyhow::Result;
use chrono::Utc;

use crate::schedule::types::{TaskType, TaskStatus};
use super::TaskManager;

pub struct TaskWorker {
    // task manager
    task_manager: Arc<TaskManager>,
    // task type. e.g. Transcribe
    task_type: TaskType,
    // interval for checking task status. e.g. 1 second
    interval: Duration,
}

impl TaskWorker {
    pub fn new(task_manager: Arc<TaskManager>, task_type: TaskType) -> Self {
        Self {
            task_manager,
            task_type,
            interval: Duration::from_secs(1),
        }
    }

    #[allow(dead_code)]
    pub fn with_interval(mut self, interval: Duration) -> Self {
        self.interval = interval;
        self
    }

    pub async fn run(&self) {
        loop {
            match self.process_next_task().await {
                Ok(true) => continue,  // continue to process next task
                Ok(false) => sleep(self.interval).await, // no task, wait
                Err(e) => {
                    error!("Error processing task: {}", e);
                    sleep(Duration::from_millis(100)).await;
                }
            }
        }
    }

    async fn process_next_task(&self) -> Result<bool> {
        // get next task to process
        let task = match self.task_manager.get_next_task().await? {
            Some(task) if task.config.task_type == self.task_type => task,
            _ => return Ok(false),
        };

        info!("Processing {} task: {}", self.task_type, task.id);

        // process task
        match self.task_manager.process_task(&task).await {
            Ok(result) => {
                // update task status and result
                let mut task = task;
                task.result = Some(result.clone());
                task.status = TaskStatus::Completed;
                task.completed_at = Some(Utc::now());
                task.updated_at = Utc::now();
                self.task_manager.storage().create(&task.clone().into()).await?;
                
                // Let the task manager handle the callback
                if let Err(e) = self.task_manager.handle_callback(&task).await {
                    error!("Failed to handle callback for task {}: {}", task.id, e);
                } else {
                    info!("Successfully processed task callback: {}", task.id);
                }
                
                Ok(true)
            }
            Err(e) => {
                error!("Failed to process task {}: {}", task.id, e);
                let mut task = task;
                task.status = TaskStatus::Failed(e.to_string());
                task.updated_at = Utc::now();
                self.task_manager.storage().create(&task.into()).await?;
                Ok(true)
            }
        }
    }
} 