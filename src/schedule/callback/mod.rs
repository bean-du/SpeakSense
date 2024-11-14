#![allow(dead_code)]

use async_trait::async_trait;
use anyhow::Result;
use serde::Serialize;
use crate::schedule::types::{Task, TaskStatus, TaskResult};
use tracing::info;
#[async_trait]
pub trait TaskCallback: Send + Sync {
    async fn on_status_change(&self, task: &Task, status: TaskStatus) -> Result<()>;
    async fn on_complete(&self, task: &Task, result: &TaskResult) -> Result<()>;
    async fn on_error(&self, task: &Task, error: &str) -> Result<()>;
    fn box_clone(&self) -> Box<dyn TaskCallback>;
}

// Implement box_clone for each type implementing TaskCallback
impl Clone for Box<dyn TaskCallback> {
    fn clone(&self) -> Self {
        self.box_clone()
    }
}

pub struct HttpCallback {
    client: reqwest::Client,
    callback_url: String,
}

#[derive(Debug, Serialize)]
struct CallbackPayload<T> {
    task_id: String,
    status: TaskStatus,
    data: T,
}

impl HttpCallback {
    pub fn new(callback_url: String) -> Self {
        Self {
            client: reqwest::Client::new(),
            callback_url,
        }
    }

    async fn send_callback<T: Serialize>(&self, payload: CallbackPayload<T>) -> Result<()> {
        info!("Sending callback to: {}", self.callback_url);
        self.client
            .post(&self.callback_url)
            .json(&payload)
            .send()
            .await?;
        Ok(())
    }

    #[allow(dead_code)]
    fn box_clone(&self) -> Box<dyn TaskCallback> {
        Box::new(Self {
            client: self.client.clone(),
            callback_url: self.callback_url.clone(),
        })
    }
}

#[async_trait]
impl TaskCallback for HttpCallback {
    async fn on_status_change(&self, task: &Task, status: TaskStatus) -> Result<()> {
        let payload = CallbackPayload {
            task_id: task.id.clone(),
            status: status.clone(),
            data: status,
        };
        self.send_callback(payload).await
    }

    fn box_clone(&self) -> Box<dyn TaskCallback> {
        Box::new(Self {
            client: self.client.clone(),
            callback_url: self.callback_url.clone(),
        })
    }

    async fn on_complete(&self, task: &Task, result: &TaskResult) -> Result<()> {
        let payload = CallbackPayload {
            task_id: task.id.clone(),
            status: TaskStatus::Completed,
            data: result,
        };
        self.send_callback(payload).await
    }

    async fn on_error(&self, task: &Task, error: &str) -> Result<()> {
        let payload = CallbackPayload {
            task_id: task.id.clone(),
            status: TaskStatus::Failed(error.to_string()),
            data: error,
        };
        self.send_callback(payload).await
    }
}

pub struct FunctionCallback<F> {
    callback: F,
}

impl<F> FunctionCallback<F>
where
    F: Fn(&Task, &str) -> Result<()> + Send + Sync + Clone + 'static,
{
    pub fn new(callback: F) -> Self {
        Self { callback }
    }

    #[allow(dead_code)]
    fn box_clone(&self) -> Box<dyn TaskCallback> {
        Box::new(Self {
            callback: self.callback.clone(),
        })
    }
}

#[async_trait]
impl<F> TaskCallback for FunctionCallback<F>
where
    F: Fn(&Task, &str) -> Result<()> + Send + Sync + Clone + 'static,
{
    async fn on_status_change(&self, task: &Task, status: TaskStatus) -> Result<()> {
        (self.callback)(task, &format!("Status changed to: {:?}", status))
    }

    async fn on_complete(&self, task: &Task, result: &TaskResult) -> Result<()> {
        (self.callback)(task, &format!("Task completed with result: {:?}", result))
    }

    async fn on_error(&self, task: &Task, error: &str) -> Result<()> {
        (self.callback)(task, &format!("Task failed: {}", error))
    }

    fn box_clone(&self) -> Box<dyn TaskCallback> {
        Box::new(Self {
            callback: self.callback.clone(),
        })
    }
}

pub struct EventCallback {
    pub sender: tokio::sync::broadcast::Sender<TaskEvent>,
}

#[derive(Debug, Clone)]
pub enum TaskEvent {
    StatusChanged { task_id: String, status: TaskStatus },
    Completed { task_id: String, result: TaskResult },
    Failed { task_id: String, error: String },
}

impl EventCallback {
    pub fn new(capacity: usize) -> (Self, tokio::sync::broadcast::Receiver<TaskEvent>) {
        let (sender, receiver) = tokio::sync::broadcast::channel(capacity);
        (Self { sender }, receiver)
    }

    #[allow(dead_code)]
    fn box_clone(&self) -> Box<dyn TaskCallback> {
        Box::new(self.clone())
    }
}

#[async_trait]
impl TaskCallback for EventCallback {
    async fn on_status_change(&self, task: &Task, status: TaskStatus) -> Result<()> {
        self.sender.send(TaskEvent::StatusChanged {
            task_id: task.id.clone(),
            status,
        })?;
        Ok(())
    }

    async fn on_complete(&self, task: &Task, result: &TaskResult) -> Result<()> {
        self.sender.send(TaskEvent::Completed {
            task_id: task.id.clone(),
            result: result.clone(),
        })?;
        Ok(())
    }

    async fn on_error(&self, task: &Task, error: &str) -> Result<()> {
        self.sender.send(TaskEvent::Failed {
            task_id: task.id.clone(),
            error: error.to_string(),
        })?;
        Ok(())
    }

    fn box_clone(&self) -> Box<dyn TaskCallback> {
        Box::new(self.clone())
    }
} 
