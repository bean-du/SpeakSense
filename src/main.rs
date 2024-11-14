#![allow(clippy::uninlined_format_args)]

use anyhow::Result;
use tracing::info;
use std::sync::Arc;
use std::net::SocketAddr;
use asr_rs::{
    asr::whisper::WhisperAsr, auth::Auth, schedule::{TaskManager, TaskScheduler}, utils::logger, AppContext, init_env, SQLITE_PATH
};
use asr_rs::storage::task::sqlite::SqliteTaskStorage;
use asr_rs::schedule::types::TaskType;
use std::fs;
use asr_rs::schedule::processors::TranscribeProcessor;
use asr_rs::storage::key::sqlite::SqliteKeyStorage;
use asr_rs::grpc::server::GrpcServer;
use asr_rs::registry::etcd::EtcdRegistry;
use asr_rs::grpc::handlers;
use asr_rs::grpc::pb::asr::asr_server;
use asr_rs::ETCD_ENDPOINT;
use asr_rs::MODEL_PATH;
use tracing::error;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize environment
    init_env();
    
    // Initialize logging system
    let _guard = logger::init("./logs".to_string())?;
    // Create necessary directories
    fs::create_dir_all("./asr_data/database")?;
    fs::create_dir_all("./asr_data/data")?;

    info!("Starting ASR service...");

    // Initialize Whisper ASR model
    info!("Initializing Whisper ASR model...");
    let asr = WhisperAsr::new(MODEL_PATH.to_string())?;
    let asr = Arc::new(asr);

    // Initialize storage
    info!("Initializing Storage...");
    let storage = SqliteTaskStorage::new(&SQLITE_PATH).await?;
    
    // Initialize auth manager
    info!("Initializing Auth Manager...");
    let api_key_storage = Arc::new(SqliteKeyStorage::new(&SQLITE_PATH).await?);
    let api_key = Arc::clone(&api_key_storage);
    let api_key_stats = Arc::clone(&api_key_storage);
    let auth_manager = Auth::new(api_key, api_key_stats);
    
    // Initialize task manager
    info!("Initializing Task Manager...");
    let mut task_manager = TaskManager::new(Arc::new(storage));


     // Register processor
     task_manager.register_processor(Box::new(TranscribeProcessor::new(Arc::clone(&asr))));

    // Create application context
    let ctx = Arc::new(AppContext {
        auth: Arc::new(auth_manager),
        task_manager: Arc::new(task_manager),
    });

   
    // Initialize scheduler and start
    info!("Initializing Scheduler...");
    let scheduler = TaskScheduler::new(ctx.task_manager.clone());
    scheduler.spawn_worker(TaskType::Transcribe).await;

    tokio::spawn(async move {
        let _ =scheduler.run().await;
    });

    // Start gRPC server
    let asr_engine = Arc::clone(&asr);
    let registry = Arc::new(EtcdRegistry::new(vec![ETCD_ENDPOINT.to_string()], 10).await?);
    let grpc_server = GrpcServer::new(registry, "asr".to_string(), "0.1.0".to_string());
    let asr_service = handlers::asr::AsrService::new(asr_engine);
    tokio::spawn(async move {
        info!("Starting gRPC server at port 7300");
        match grpc_server.start(asr_server::AsrServer::new(asr_service), 7300).await {
            Ok(_) => info!("gRPC server started successfully"),
            Err(e) => error!("Failed to start gRPC server: {}", e),
        }
    });

    // Start HTTP server
    let addr = SocketAddr::from(([0, 0, 0, 0], 7200));
    match asr_rs::web::start_server(ctx.clone(), addr).await {
        Ok(_) => info!("Server stopped gracefully"),
        Err(e) => {
            tracing::error!("Server error: {}", e);
            return Err(e);
        }
    }

    // Graceful shutdown
    info!("Shutting down...");
    Ok(())
}

