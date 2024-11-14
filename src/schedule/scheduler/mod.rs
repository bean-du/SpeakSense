mod task_manager;
mod worker;

use std::sync::Arc;
use tokio::task::JoinHandle;
use tokio::sync::Mutex;
use anyhow::Result;

pub use task_manager::TaskManager;
use worker::TaskWorker;
use crate::schedule::types::TaskType;

pub struct TaskScheduler {
    task_manager: Arc<TaskManager>,
    workers: Mutex<Vec<JoinHandle<()>>>,
}

impl TaskScheduler {
    pub fn new(task_manager: Arc<TaskManager>) -> Self {
        Self {
            task_manager,
            workers: Mutex::new(Vec::new()),
        }
    }

    pub async fn spawn_worker(&self, task_type: TaskType) {
        let worker = TaskWorker::new(self.task_manager.clone(), task_type);
        let handle = tokio::spawn(async move {
            worker.run().await;
        });
        self.workers.lock().await.push(handle);
    }

    pub async fn run(&self) -> Result<()> {
        // start task timeout check
        let tm = self.task_manager.clone();
        tokio::spawn(async move {
            loop {
                if let Err(e) = tm.handle_timed_out_tasks().await {
                    tracing::error!("Error handling timed out tasks: {}", e);
                }
                tokio::time::sleep(std::time::Duration::from_secs(60)).await;
            }
        });

        // wait for all workers to finish
        let mut workers = self.workers.lock().await;
        for worker in workers.drain(..) {
            worker.await?;
        }

        Ok(())
    }
} 