use axum::Router;
use std::sync::Arc;
use crate::AppContext;

pub mod asr;
pub mod auth;
pub mod schedule;
pub mod callback_test;

pub fn router(ctx: Arc<AppContext>) -> Router {
    Router::new()
        .nest("/asr", asr::transcribe_router(ctx.clone()))
        .nest("/auth", auth::auth_router(ctx.auth.clone()))
        .nest("/schedule", schedule::schedule_router(ctx.task_manager.clone()))
        .nest("/callback", callback_test::callback_router())
} 