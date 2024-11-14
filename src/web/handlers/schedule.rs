use axum::{
    routing::{post, get},
    Router,
    extract::{State, Path, Json},
    response::IntoResponse,
    http::StatusCode,
};
use std::sync::Arc;
use serde::{Deserialize, Serialize};

use crate::web::Pagination;
use crate::schedule::types::{TaskConfig,  TaskPriority};
use crate::schedule::scheduler::TaskManager;
use tracing::error;

pub fn schedule_router(task_manager: Arc<TaskManager>) -> Router {
    Router::new()
        .route("/tasks", post(create_task))
        .route("/tasks/:task_id", get(get_task))
        .route("/tasks/:task_id/status", get(get_task_status))
        .route("/tasks/:task_id/priority", post(update_task_priority))
        .route("/tasks/stats", get(get_task_stats))
        .with_state(task_manager)
}

#[derive(Debug, Serialize)]
struct ApiResponse<T> {
    success: bool,
    data: Option<T>,
    error: Option<String>,
}

impl<T: Serialize> ApiResponse<T> {
    fn success(data: T) -> Self {
        Self {
            success: true,
            data: Some(data),
            error: None,
        }
    }

    fn error(error: String) -> Self {
        Self {
            success: false,
            data: None,
            error: Some(error),
        }
    }
}

// Create task endpoint
async fn create_task(
    State(task_manager): State<Arc<TaskManager>>,
    Json(config): Json<TaskConfig>,
) -> impl IntoResponse {
    match task_manager.create_task(config).await {
        Ok(task) => (
            StatusCode::CREATED,
            Json(ApiResponse::success(task))
        ),
        Err(e) => {
            error!("Failed to create task: {}", e);
            (
                StatusCode::BAD_REQUEST,
                Json(ApiResponse::error(e.to_string()))
            )
        },
    }
}

// Get task endpoint
async fn get_task(
    State(task_manager): State<Arc<TaskManager>>,
    Path(task_id): Path<String>,
) -> impl IntoResponse {
    match task_manager.get_task(&task_id).await {
        Ok(Some(task)) => (
            StatusCode::OK,
            Json(ApiResponse::success(task))
        ),
        Ok(None) => (
            StatusCode::NOT_FOUND,
            Json(ApiResponse::error("Task not found".to_string()))
        ),
        Err(e) => {
            error!("Failed to get task: {}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ApiResponse::error(e.to_string()))
            )
        },
    }
}

// Get task status endpoint
async fn get_task_status(
    State(task_manager): State<Arc<TaskManager>>,
    Path(task_id): Path<String>,
) -> impl IntoResponse {
    match task_manager.get_task_status(&task_id).await {
        Ok(Some(status)) => (
            StatusCode::OK,
            Json(ApiResponse::success(status))
        ),
        Ok(None) => (
            StatusCode::NOT_FOUND,
            Json(ApiResponse::error("Task not found".to_string()))
        ),
        Err(e) => {
            error!("Failed to get task status: {}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ApiResponse::error(e.to_string()))
            )
        },
    }
}

#[derive(Debug, Deserialize)]
struct UpdatePriorityRequest {
    priority: TaskPriority,
}

// Update task priority endpoint
async fn update_task_priority(
    State(task_manager): State<Arc<TaskManager>>,
    Path(task_id): Path<String>,
    Json(req): Json<UpdatePriorityRequest>,
) -> impl IntoResponse {
    match task_manager.update_task_priority(&task_id, req.priority).await {
        Ok(_) => (
            StatusCode::OK,
            Json(ApiResponse::<()>::success(()))
        ),
        Err(e) => {
            error!("Failed to update task priority: {}", e);
            (
                StatusCode::BAD_REQUEST,
                Json(ApiResponse::error(e.to_string()))
            )
        },
    }
}

// Get task stats endpoint
async fn get_task_stats(
    State(task_manager): State<Arc<TaskManager>>,
    Path(pagination): Path<Pagination>,
) -> impl IntoResponse {
    match task_manager.get_task_stats(&pagination).await {
        Ok(stats) => (
            StatusCode::OK,
            Json(ApiResponse::success(stats)),
        ),
            Err(e) => {
            error!("Failed to get task stats: {}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ApiResponse::error(e.to_string()))
            )
        },
    }
} 