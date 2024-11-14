use axum::Router;
use axum::routing::post;
use axum::extract::Json;
use tracing::info;

pub fn callback_router() -> Router {
    Router::new()
        .route("/http", post(http_callback))
}

async fn http_callback(Json(payload): Json<serde_json::Value>) {
    info!("Received callback: {:?}", payload);
}