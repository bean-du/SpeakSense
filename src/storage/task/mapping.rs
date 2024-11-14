use crate::storage::task::entity::Model as TaskModel;
use crate::schedule::types::Task;

impl From<TaskModel> for Task {
    fn from(model: TaskModel) -> Self {
        Task {
            id: model.id,
            status: serde_json::from_str(&model.status).unwrap(),
            config: serde_json::from_str(&model.config).unwrap(),
            created_at: model.created_at,
            updated_at: model.updated_at,
            started_at: model.started_at,
            completed_at: model.completed_at,
            result: model.result.map(|r| serde_json::from_str(&r).unwrap()),
            error: model.error,
        }
    }
}

impl From<Task> for TaskModel {
    fn from(task: Task) -> Self {
        TaskModel {
            id: task.id,
            status: serde_json::to_string(&task.status).unwrap(),
            config: serde_json::to_string(&task.config).unwrap(),
            created_at: task.created_at,
            updated_at: task.updated_at,
            started_at: task.started_at,
            completed_at: task.completed_at,
            result: task.result.map(|r| serde_json::to_string(&r).unwrap()),
            error: task.error,
            priority: task.config.priority as i32,
            retry_count: task.config.retry_count as i32,
            max_retries: task.config.max_retries as i32,
            timeout: task.config.timeout.map(|t| t as i64),
        }
    }
}


