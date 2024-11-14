mod key_entity;
pub mod sqlite;
mod key_stats_entity;


mod tests;


pub use key_entity::Model as ApiKey;
pub use sqlite::SqliteKeyStorage;
pub use crate::auth::types::KeyStatus;
use crate::auth::ApiKeyInfo;
use anyhow::Result;
use crate::auth::stats::ApiKeyStats;
use async_trait::async_trait;   
use chrono::Utc;
use std::collections::HashMap;

#[async_trait]
pub trait KeyStorage: Send + Sync + 'static {
    async fn get(&self, api_key: &str) -> Result<Option<ApiKey>>;
    async fn create(&self, api_key: String, info: ApiKey) -> Result<()>;
    async fn remove(&self, api_key: &str) -> Result<()>;
    async fn list(&self) -> Result<Vec<ApiKey>>;
    async fn update_status(&self, api_key: &str, status: KeyStatus) -> Result<()>;
}


#[async_trait]
pub trait ApiKeyStatsStorage: Send + Sync + 'static {
    async fn get_stats(&self, api_key: &str) -> Result<Option<ApiKeyStats>>;
    async fn update_stats(&self, api_key: &str, stats: ApiKeyStats) -> Result<()>;
}

impl From<ApiKey> for ApiKeyInfo {
    fn from(key: ApiKey) -> Self {
        let permissions = key.get_permissions();
        let rate_limit = key.get_rate_limit();
        let status = serde_json::from_str(&key.status)
            .unwrap_or(KeyStatus::Expired);

        ApiKeyInfo {
            key: key.key,
            name: key.name,
            created_at: key.created_at,
            expires_at: key.expires_at,
            permissions,
            rate_limit,
            status,
        }
    }
}

impl From<ApiKeyInfo> for ApiKey {
    fn from(info: ApiKeyInfo) -> Self {
        let mut key = ApiKey {
            key: info.key,
            name: info.name,
            created_at: info.created_at,
            expires_at: info.expires_at,
            permissions: String::new(),
            rate_limit: String::new(),
            status: serde_json::to_string(&info.status)
                .unwrap_or_else(|_| "inactive".to_string()),
        };
        key.set_permissions(info.permissions);
        key.set_rate_limit(info.rate_limit);
        key
    }
}

impl From<ApiKeyInfo> for key_entity::ActiveModel {
    fn from(info: ApiKeyInfo) -> Self {
        use sea_orm::ActiveValue::Set;
        
        key_entity::ActiveModel {
            key: Set(info.key),
            name: Set(info.name),
            created_at: Set(info.created_at),
            expires_at: Set(info.expires_at),
            permissions: Set(serde_json::to_string(&info.permissions)
                .unwrap_or_default()),
            rate_limit: Set(serde_json::to_string(&info.rate_limit)
                .unwrap_or_default()),
            status: Set(serde_json::to_string(&info.status)
                .unwrap_or_else(|_| "inactive".to_string())),
        }
    }
}

impl From<key_stats_entity::Model> for ApiKeyStats {
    fn from(model: key_stats_entity::Model) -> Self {
        let mut requests_per_day = HashMap::new();
        requests_per_day.insert(Utc::now().date_naive().to_string(), model.requests_today as u64);
        
        ApiKeyStats {
            total_requests: model.total_requests as u64,
            requests_today: model.requests_today as u64,
            last_used_at: model.last_request_at.unwrap_or_else(|| Utc::now()),
            requests_per_day,
        }
    }
}

impl From<ApiKeyStats> for key_stats_entity::ActiveModel {
    fn from(stats: ApiKeyStats) -> Self {
        use sea_orm::ActiveValue::Set;
        
        key_stats_entity::ActiveModel {
            key: Set(stats.last_used_at.to_string()), 
            requests_today: Set(stats.requests_today as i64),
            requests_this_hour: Set(0),  
            requests_this_minute: Set(0),
            last_request_at: Set(Some(stats.last_used_at)),
            total_requests: Set(stats.total_requests as i64),
            total_tokens: Set(0),
            total_audio_seconds: Set(0),
            updated_at: Set(Utc::now()),
        }
    }
}