#![warn(dead_code)]

use axum::{
    extract::State,
    http::StatusCode,
    Json,
    extract::Path,
    Router,
    routing::{post, delete, get},
    response::IntoResponse,
};
use serde::{Deserialize, Serialize};
use crate::Auth;
use crate::auth::{Permission, RateLimit, ApiKeyInfo};

use std::sync::Arc;

pub fn auth_router(auth: Arc<Auth>) -> Router {
    Router::new()
        .route("/api-keys", post(create_api_key))
        .route("/api-keys/:api_key", delete(revoke_api_key))
        .route("/api-keys/:api_key/stats", get(get_key_stats))
        .route("/api-keys/:api_key/usage", get(get_key_usage_report))
        .with_state(auth)
}

#[derive(Debug, Deserialize)]
pub struct CreateApiKeyRequest {
    pub name: String,
    pub permissions: Vec<Permission>,
    pub rate_limit: RateLimit,
    pub expires_in_days: Option<i64>,
}

#[derive(Debug, Serialize)]
pub struct ApiKeyResponse {
    pub key_info: ApiKeyInfo,
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

async fn create_api_key(
    State(auth): State<Arc<Auth>>,
    Json(req): Json<CreateApiKeyRequest>,
) -> impl IntoResponse {
    match auth.create_api_key(
        req.name,
        req.permissions,
        req.rate_limit,
        req.expires_in_days,
    ).await {
        Ok(key_info) => (
            StatusCode::CREATED,
            Json(ApiResponse::success(ApiKeyResponse { key_info }))
        ),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ApiResponse::error(e.to_string()))
        ),
    }
}

async fn revoke_api_key(
    State(auth): State<Arc<Auth>>,
    Path(api_key): Path<String>,
) -> impl IntoResponse {
    match auth.revoke_api_key(&api_key).await {
        Ok(_) => (
            StatusCode::OK,
            Json(ApiResponse::<()>::success(()))
        ),
        Err(e) => (
            StatusCode::NOT_FOUND,
            Json(ApiResponse::error(e.to_string()))
        ),
    }
}

async fn get_key_stats(
    State(auth): State<Arc<Auth>>,
    Path(api_key): Path<String>,
) -> impl IntoResponse {
    match auth.get_key_stats(&api_key).await {
        Ok(stats) => (
            StatusCode::OK,
            Json(ApiResponse::success(stats))
        ),
        Err(e) => (
            StatusCode::NOT_FOUND,
            Json(ApiResponse::error(e.to_string()))
        ),
    }
}

async fn get_key_usage_report(
    State(auth): State<Arc<Auth>>,
    Path(api_key): Path<String>,
) -> impl IntoResponse {
    match auth.get_key_usage_report(&api_key).await {
        Ok(report) => (
            StatusCode::OK,
            Json(ApiResponse::success(report))
        ),
        Err(e) => (
            StatusCode::NOT_FOUND,
            Json(ApiResponse::error(e.to_string()))
        ),
    }
} 