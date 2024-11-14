pub mod asr;
pub mod auth;
pub mod schedule;
pub mod utils;
pub mod web;
pub mod storage;
pub mod audio;
pub mod grpc;
pub mod registry;

use std::{env, sync::Arc};
use auth::Auth;
use schedule::TaskManager;
use once_cell::sync::Lazy;

pub struct AppContext {
    pub auth: Arc<Auth>,
    pub task_manager: Arc<TaskManager>,
}

const ASR_SQLITE_PATH: &str = "sqlite://./asr_data/database/storage.db?mode=rwc";
const ASR_AUDIO_PATH: &str = "./asr_data/audio/";
const ETCD_DEFAULT_ENDPOINT: &str = "http://localhost:2379";
const ASR_MODEL_PATH: &str = "./models/ggml-large-v3.bin";

pub static MODEL_PATH: Lazy<String> = Lazy::new(|| {
    match env::var("ASR_MODEL_PATH") {
        Ok(path) => path,
        Err(_) => {
            dotenv::var("ASR_MODEL_PATH").unwrap_or_else(|_| ASR_MODEL_PATH.to_string())
        }
    }
});

pub static SQLITE_PATH: Lazy<String> = Lazy::new(|| {
    match env::var("ASR_SQLITE_PATH") {
        Ok(path) => path,
        Err(_) => {
            dotenv::var("ASR_SQLITE_PATH").unwrap_or_else(|_| ASR_SQLITE_PATH.to_string())
        }
    }
});

pub static ETCD_ENDPOINT: Lazy<String> = Lazy::new(|| {
    match env::var("ETCD_ENDPOINT") {
        Ok(path) => path,
        Err(_) => {
            dotenv::var("ETCD_ENDPOINT").unwrap_or_else(|_| ETCD_DEFAULT_ENDPOINT.to_string())
        }
    }
});

pub static AUDIO_PATH: Lazy<String> = Lazy::new(|| {
    match env::var("ASR_AUDIO_PATH") {
        Ok(path) => path,
        Err(_) => {
            dotenv::var("ASR_AUDIO_PATH").unwrap_or_else(|_| ASR_AUDIO_PATH.to_string())
        }
    }
});

pub fn init_env() {
    dotenv::dotenv().ok();
    
    // ensure database directory exists
    if let Some(db_path) = SQLITE_PATH.strip_prefix("sqlite://") {
        if let Some(dir) = std::path::Path::new(db_path).parent() {
            std::fs::create_dir_all(dir).unwrap_or_else(|e| {
                eprintln!("Failed to create database directory: {}", e);
            });
        }
    }
}