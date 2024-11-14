use async_trait::async_trait;
use anyhow::Result;
use chrono::{DateTime, Utc};
use crate::storage::task::entity::Model as TaskModel;
use crate::web::Pagination;
pub mod sqlite;
pub mod entity;
pub mod mapping;

#[async_trait]
pub trait TaskStorage: Send + Sync + 'static {
    async fn create(&self, model: &TaskModel) -> Result<()>;
    async fn list(&self, pagination: &Pagination) -> Result<Vec<TaskModel>>;
    async fn get_pending_by_priority(&self, limit: usize) -> Result<Vec<TaskModel>>;
    async fn get(&self, task_id: &str) -> Result<Option<TaskModel>>;
    async fn update(&self, task_id: &str, status: &str) -> Result<()>;
    async fn delete(&self, task_id: &str) -> Result<()>;
    async fn get_timeouted(&self) -> Result<Vec<TaskModel>>;
    async fn cleanup_old(&self, before: DateTime<Utc>) -> Result<u64>;
    async fn get_by_status(&self, status: &str) -> Result<Vec<TaskModel>>;
}

#[cfg(test)]
mod tests;