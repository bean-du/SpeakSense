use super::*;
use crate::schedule::types::{
    TaskType, CallbackType, TaskParams, TranscribeParams, 
    TaskStatus, TaskConfig, TaskPriority
};
use chrono::Duration;
use tempfile::NamedTempFile;
use uuid::Uuid;
use std::path::PathBuf;
use crate::storage::task::sqlite::SqliteTaskStorage;
use crate::schedule::types::Task;
use crate::storage::task::entity::Model as TaskModel;
use crate::web::handlers::asr::PathType;
use crate::SQLITE_PATH;

async fn setup_storage() -> (SqliteTaskStorage, NamedTempFile) {
    let temp_file = NamedTempFile::new().unwrap();
    let storage = SqliteTaskStorage::new(&SQLITE_PATH).await.unwrap();
    (storage, temp_file)
}

fn create_test_task(priority: TaskPriority) -> Task {
    Task {
        id: Uuid::new_v4().to_string(),
        status: TaskStatus::Pending,
        config: TaskConfig {
            task_type: TaskType::Transcribe,
            path_type: PathType::Url,
            callback_type: CallbackType::Http { url: "http://localhost:3000/callback".to_string() },
            params: TaskParams::Transcribe(TranscribeParams {
                language: None,
                speaker_diarization: false,
                emotion_recognition: false,
                filter_dirty_words: false,
            }),
            input_path: PathBuf::from("/path/to/input").to_str().unwrap().to_string(),
            priority,
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
    }
}

#[tokio::test]
async fn test_save_and_get_task() {
    let (storage, _temp_file) = setup_storage().await;
    let task = create_test_task(TaskPriority::Normal);
    
    let model = TaskModel::from(task.clone());
    storage.create(&model).await.unwrap();
    let retrieved_model = storage.get(&task.id).await.unwrap().unwrap();
    let retrieved_task = Task::from(retrieved_model);
    
    assert_eq!(task.id, retrieved_task.id);
    assert_eq!(task.status, retrieved_task.status);
}

#[tokio::test]
async fn test_get_pending_tasks_priority_order() {
    let (storage, _temp_file) = setup_storage().await;
    
    let task1 = create_test_task(TaskPriority::Low);
    let task2 = create_test_task(TaskPriority::High);
    let task3 = create_test_task(TaskPriority::Normal);
    
    storage.create(&TaskModel::from(task1.clone())).await.unwrap();
    storage.create(&TaskModel::from(task2.clone())).await.unwrap();
    storage.create(&TaskModel::from(task3)).await.unwrap();
    
    let pending_models = storage.get_pending_by_priority(10).await.unwrap();
    let pending_tasks: Vec<Task> = pending_models.into_iter().map(Task::from).collect();
    
    assert_eq!(pending_tasks.len(), 3);
    assert_eq!(pending_tasks[0].config.priority, TaskPriority::High);
    assert_eq!(pending_tasks[2].config.priority, TaskPriority::Low);
}

#[tokio::test]
async fn test_update_task_status() {
    let (storage, _temp_file) = setup_storage().await;
    let task = create_test_task(TaskPriority::Normal);
    
    storage.create(&TaskModel::from(task.clone())).await.unwrap();
    storage.update(&task.id, &serde_json::to_string(&TaskStatus::Processing).unwrap()).await.unwrap();
    
    let updated_model = storage.get(&task.id).await.unwrap().unwrap();
    let updated_task = Task::from(updated_model);
    assert_eq!(updated_task.status, TaskStatus::Processing);
    assert!(updated_task.started_at.is_some());
}

#[tokio::test]
async fn test_delete_task() {
    let (storage, _temp_file) = setup_storage().await;
    let task = create_test_task(TaskPriority::Normal);
    
    storage.create(&TaskModel::from(task.clone())).await.unwrap();
    storage.delete(&task.id).await.unwrap();
    
    let result = storage.get(&task.id).await.unwrap();
    assert!(result.is_none());
}

#[tokio::test]
async fn test_get_timed_out_tasks() {
    let (storage, _temp_file) = setup_storage().await;
    let mut task = create_test_task(TaskPriority::Normal);
    task.status = TaskStatus::Processing;
    task.started_at = Some(Utc::now() - Duration::seconds(301));
    
    storage.create(&TaskModel::from(task.clone())).await.unwrap();
    
    let timed_out_models = storage.get_timeouted().await.unwrap();
    let timed_out_tasks: Vec<Task> = timed_out_models.into_iter().map(Task::from).collect();
    assert_eq!(timed_out_tasks.len(), 1);
    assert_eq!(timed_out_tasks[0].id, task.id);
}

#[tokio::test]
async fn test_cleanup_old_tasks() {
    let (storage, _temp_file) = setup_storage().await;
    let mut task = create_test_task(TaskPriority::Normal);
    task.status = TaskStatus::Completed;
    task.updated_at = Utc::now() - Duration::hours(25);
    
    storage.create(&TaskModel::from(task.clone())).await.unwrap();
    
    let cleaned = storage.cleanup_old(Utc::now() - Duration::hours(24)).await.unwrap();
    assert_eq!(cleaned, 1);
    
    let result = storage.get(&task.id).await.unwrap();
    assert!(result.is_none());
}

#[tokio::test]
async fn test_get_tasks_by_status() {
    let (storage, _temp_file) = setup_storage().await;
    let mut task = create_test_task(TaskPriority::Normal);
    let status = TaskStatus::Failed("Test failure".to_string());
    task.status = status.clone();
    
    storage.create(&TaskModel::from(task.clone())).await.unwrap();
    
    let status_str = serde_json::to_string(&status).unwrap();
    let failed_models = storage.get_by_status(&status_str).await.unwrap();
    let failed_tasks: Vec<Task> = failed_models.into_iter().map(Task::from).collect();
    assert_eq!(failed_tasks.len(), 1);
    assert_eq!(failed_tasks[0].id, task.id);
} 