#![allow(dead_code)]

use std::sync::Arc;
use std::collections::HashMap;
use tokio::sync::Mutex;
use anyhow::Result;
use uuid::Uuid;
use chrono::{DateTime, Utc};
use tracing::{info, warn, error};
use serde::{Serialize, Deserialize};

use crate::schedule::types::{
    Task, TaskConfig, TaskResult, TaskStatus, TaskType,
    CallbackType, TaskPriority
};
use crate::storage::task::TaskStorage;
use crate::schedule::processors::TaskProcessor;
use crate::schedule::callback::{
    TaskCallback, HttpCallback, FunctionCallback, EventCallback,
};
use crate::web::Pagination;

pub struct TaskManager {
    pub storage: Arc<dyn TaskStorage>,
    processors: HashMap<TaskType, Box<dyn TaskProcessor>>,
    processing_tasks: Mutex<HashMap<String, ProcessingInfo>>,
    function_callbacks: HashMap<String, Box<dyn TaskCallback>>,
    event_callback: EventCallback,
}

#[derive(Debug)]
struct ProcessingInfo {
    #[allow(dead_code)]
    status: TaskStatus,
    started_at: DateTime<Utc>,
    attempts: u32,
}

impl TaskManager {
    pub fn new(storage: Arc<dyn TaskStorage>) -> Self {
        let (event_callback, _) = EventCallback::new(10);
        Self {
            storage,
            processors: HashMap::new(),
            processing_tasks: Mutex::new(HashMap::new()),
            function_callbacks: HashMap::new(),
            event_callback,
        }
    }

    pub fn storage(&self) -> &Arc<dyn TaskStorage> {
        &self.storage
    }

    pub fn register_processor(&mut self, processor: Box<dyn TaskProcessor>) {
        let task_type = processor.task_type();
        info!("Registering processor for task type: {:?}", task_type);
        self.processors.insert(task_type, processor);
    }

    pub async fn create_task(&self, config: TaskConfig) -> Result<Task> {
        // validate task params
        let processor = self.processors.get(&config.task_type)
            .ok_or_else(|| anyhow::anyhow!("No processor found for task type: {:?}", config.task_type))?;
        
        processor.validate_params(&config.params)?;

        let task = Task {
            id: format!("task-{}", Uuid::new_v4()),
            status: TaskStatus::Pending,
            config,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            started_at: None,
            completed_at: None,
            result: None,
            error: None,
        };

        self.storage.create(&task.clone().into()).await?;
        info!("Creating new task: {}", task.id);
        Ok(task)
    }

    pub async fn get_next_task(&self) -> Result<Option<Task>> {
        let mut processing = self.processing_tasks.lock().await;
        
        // clean up stale processing tasks
        self.cleanup_stale_tasks(&mut processing).await?;
        
        // get pending tasks by priority
        let pending_tasks = self.storage.get_pending_by_priority(10).await?;
        
        // process tasks by priority
        for task in pending_tasks {
            if !processing.contains_key(&task.id) {
                info!("Starting task {}", task.id);
                
                // mark task as processing
                processing.insert(task.id.clone(), ProcessingInfo {
                    status: TaskStatus::Processing,
                    started_at: Utc::now(),
                    attempts: 1,
                });
                
                // update task status in database
                self.storage.update(&task.id, &TaskStatus::Processing.to_string()).await?;
                
                // record task start time
                let mut task = task;
                task.started_at = Some(Utc::now());
                self.storage.create(&task.clone().into()).await?;
                
                return Ok(Some(task.into()));
            }
        }

        Ok(None)
    }

    pub async fn process_task(&self, task: &Task) -> Result<TaskResult> {
        let processor = self.processors.get(&task.config.task_type)
            .ok_or_else(|| anyhow::anyhow!("No processor found for task type"))?;

        info!("Processing task {} with processor {:?}", task.id, task.config.task_type);
        
        match processor.process(task).await {
            Ok(result) => {
                info!("Task {} completed successfully", task.id);
                Ok(result)
            }
            Err(e) => {
                error!("Failed to process task {}: {}", task.id, e);
                self.handle_task_error(task, e).await?;
                Err(anyhow::anyhow!("Task processing failed"))
            }
        }
    }

    async fn handle_task_error(&self, task: &Task, error: anyhow::Error) -> Result<()> {
        let mut processing = self.processing_tasks.lock().await;
        
        if let Some(info) = processing.get_mut(&task.id) {
            if info.attempts < task.config.max_retries {
                info.attempts += 1;
                warn!("Retrying task {} (attempt {}/{})", task.id, info.attempts, task.config.max_retries);
                
                self.storage.update(&task.id, &TaskStatus::Retrying.to_string()).await?;
            } else {
                error!("Task {} failed after {} attempts", task.id, info.attempts);
                
                self.storage.update(&task.id, &TaskStatus::Failed(error.to_string()).to_string()).await?;
                
                processing.remove(&task.id);
            }
        }
        
        Ok(())
    }

    async fn cleanup_stale_tasks(&self, processing: &mut HashMap<String, ProcessingInfo>) -> Result<()> {
        let now = Utc::now();
        let mut to_remove = Vec::new();

        for (task_id, info) in processing.iter() {
            let duration = now - info.started_at;
            if duration.num_minutes() > 30 { // 设置30分钟超时
                to_remove.push(task_id.clone());
                warn!("Task {} timed out after {} minutes", task_id, duration.num_minutes());
            }
        }

        for task_id in to_remove {
            processing.remove(&task_id);
            self.storage.update(&task_id, &TaskStatus::TimedOut.to_string()).await?;
        }

        Ok(())
    }

    // task status query method
    pub async fn get_task_status(&self, task_id: &str) -> Result<Option<TaskStatus>> {
        Ok(self.storage.get(task_id).await?.map(|t| TaskStatus::try_from(t.status).unwrap()))
    }

    // task stats method
    pub async fn get_task_stats(&self, pagination: &Pagination) -> Result<TaskStats> {
        let all_tasks = self.storage.list(pagination).await?;
        let mut stats = TaskStats::default();

        for model in all_tasks {
            let task = Task::from(model);
            match task.status {
                TaskStatus::Pending => stats.pending += 1,
                TaskStatus::Processing => stats.processing += 1,
                TaskStatus::Completed => stats.completed += 1,
                TaskStatus::Failed(_) => stats.failed += 1,
                TaskStatus::Retrying => stats.retrying += 1,
                TaskStatus::TimedOut => stats.timed_out += 1,
            }
        }

        Ok(stats)
    }

    // task cleanup method
    pub async fn cleanup_tasks(&self, retention_days: i64) -> Result<CleanupStats> {
        let cutoff = Utc::now() - chrono::Duration::days(retention_days);
        let mut stats = CleanupStats::default();

        // clean up completed tasks
        stats.completed = self.storage.cleanup_old(cutoff).await?;

        // clean up failed tasks
        let failed_tasks = self.storage.get_by_status(&TaskStatus::Failed("".into()).to_string()).await?;
        for task in failed_tasks {
            if task.updated_at < cutoff {
                self.storage.delete(&task.id).await?;
                stats.failed += 1;
            }
        }

        Ok(stats)
    }

    pub async fn handle_callback(&self, task: &Task) -> Result<()> {
        // handle callback by callback type and complete status change
        match &task.config.callback_type {
            CallbackType::Http { url } => {
                let callback = HttpCallback::new(url.clone());
                match task.status {
                    TaskStatus::Completed => callback.on_complete(task, &task.result.clone().unwrap()).await?,
                    TaskStatus::Failed(ref error) => callback.on_error(task, error).await?,
                    _ => return Ok(()),
                }
            }
            CallbackType::Function { name } => {
                let callback = self.get_function_callback(name)?;
                match task.status {
                    TaskStatus::Completed => callback.on_complete(task, &task.result.clone().unwrap()).await?,
                    TaskStatus::Failed(ref error) => callback.on_error(task, error).await?,
                    _ => return Ok(()),
                }
            }
            CallbackType::Event => {
                let callback = self.event_callback.clone();
                match task.status {
                    TaskStatus::Completed => callback.on_complete(task, &task.result.clone().unwrap()).await?,
                    TaskStatus::Failed(ref error) => callback.on_error(task, error).await?,
                    _ => return Ok(()),
                }
            }
            CallbackType::None => return Ok(()),
        }
        Ok(())
    }

    pub fn register_function_callback<F>(&mut self, name: &str, callback: F)
    where
        F: Fn(&Task, &str) -> Result<()> + Send + Sync + Clone + 'static,
    {
        self.function_callbacks.insert(
            name.to_string(),
            Box::new(FunctionCallback::new(callback)),
        );
    }

    fn get_function_callback(&self, name: &str) -> Result<Box<dyn TaskCallback>> {
        self.function_callbacks
            .get(name)
            .map(|cb| cb.box_clone())
            .ok_or_else(|| anyhow::anyhow!("Callback function not found: {}", name))
    }

    pub async fn handle_timed_out_tasks(&self) -> Result<()> {
        let timed_out_tasks = self.storage.get_timeouted().await?;
        
        for task in timed_out_tasks {
            info!("Handling timed out task: {}", task.id);
            self.storage.update(&task.id, &TaskStatus::TimedOut.to_string()).await?;
        }
        
        Ok(())
    }

    // get task method
    pub async fn get_task(&self, task_id: &str) -> Result<Option<Task>> {
        let model = self.storage.get(task_id).await?;
        Ok(model.map(|m| Task::from(m)))
    }

    // update task priority method
    pub async fn update_task_priority(&self, task_id: &str, new_priority: TaskPriority) -> Result<()> {
        let model = self.storage.get(task_id).await?
            .ok_or_else(|| anyhow::anyhow!("Task not found"))?;
        let task = Task::from(model);

        // only allow to adjust priority of pending tasks
        if task.status != TaskStatus::Pending {
            return Err(anyhow::anyhow!("Can only adjust priority of pending tasks"));
        }
        
        let mut task = task.clone();
        task.config.priority = new_priority;
        task.updated_at = Utc::now();
        
        self.storage.create(&task.clone().into()).await
    }

    // get timed out tasks method
    pub async fn get_timed_out_tasks(&self) -> Result<Vec<Task>> {
        self.storage.get_timeouted().await.map(|models| models.into_iter().map(|m| Task::from(m)).collect())
    }
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct TaskStats {
    pub pending: usize,
    pub processing: usize,
    pub completed: usize,
    pub failed: usize,
    pub retrying: usize,
    pub timed_out: usize,
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct CleanupStats {
    pub completed: u64,
    pub failed: u64,
}

// implement Drop trait for TaskManager to ensure resources are cleaned up correctly
impl Drop for TaskManager {
    fn drop(&mut self) {
        info!("TaskManager is being dropped, cleaning up resources...");
    }
}

impl Clone for EventCallback {
    fn clone(&self) -> Self {
        let (sender, _) = tokio::sync::broadcast::channel(10);
        Self { sender }
    }
} 