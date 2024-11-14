use sea_orm::*;
use sea_query::OnConflict;
use anyhow::Result;
use super::key_entity::{self, Model as ApiKey};
use crate::storage::Db;
use super::KeyStorage;
use crate::auth::types::KeyStatus;
use super::key_stats_entity;
use crate::storage::key::ApiKeyStatsStorage;
use crate::auth::stats::ApiKeyStats;
use async_trait::async_trait;
use tracing::info;

#[derive(Clone)]
pub struct SqliteKeyStorage {
    db: Db,
}

impl SqliteKeyStorage {
    pub async fn new(database_url: &str) -> Result<Self> {
        info!("Initializing SQLite API key storage at {}", database_url);

        let db = Database::connect(
            ConnectOptions::new(database_url.to_owned())
                .sqlx_logging(false)
                .to_owned()
        ).await?;

        db.execute(Statement::from_string(
            DbBackend::Sqlite,
            r#"
            CREATE TABLE IF NOT EXISTS api_keys (
                key TEXT PRIMARY KEY NOT NULL,
                name TEXT NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT,
                permissions TEXT NOT NULL,
                rate_limit TEXT NOT NULL,
                status TEXT NOT NULL
            )
            "#.to_owned(),
        ))
        .await?;

        db.execute(Statement::from_string(
            DbBackend::Sqlite,
            r#"
            CREATE TABLE IF NOT EXISTS api_key_stats (
                key TEXT PRIMARY KEY NOT NULL,
                requests_today INTEGER NOT NULL DEFAULT 0,
                requests_this_hour INTEGER NOT NULL DEFAULT 0,
                requests_this_minute INTEGER NOT NULL DEFAULT 0,
                last_request_at TEXT,
                total_requests INTEGER NOT NULL DEFAULT 0,
                total_tokens INTEGER NOT NULL DEFAULT 0,
                total_audio_seconds INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL
            )
            "#.to_owned(),
        ))
        .await?;

        Ok(Self { db })
    }
}

#[async_trait]
impl KeyStorage for SqliteKeyStorage {
    async fn get(&self, api_key: &str) -> Result<Option<ApiKey>> {
        let key = key_entity::Entity::find()
            .filter(key_entity::Column::Key.eq(api_key))
            .one(&self.db)
            .await?;
        Ok(key)
    }

    async fn create(&self, api_key: String, info: ApiKey) -> Result<()> {
        let active_model = key_entity::ActiveModel {
            key: Set(api_key),
            name: Set(info.name.clone()),
            created_at: Set(info.created_at),
            expires_at: Set(info.expires_at),
            permissions: Set(serde_json::to_string(&info.get_permissions())?),
            rate_limit: Set(serde_json::to_string(&info.get_rate_limit())?),
            status: Set(info.status),
        };
        
        key_entity::Entity::insert(active_model)
            .exec(&self.db)
            .await?;
        Ok(())
    }

    async fn remove(&self, api_key: &str) -> Result<()> {
        key_entity::Entity::delete_many()
            .filter(key_entity::Column::Key.eq(api_key))
            .exec(&self.db)
            .await?;
        Ok(())
    }

    async fn list(&self) -> Result<Vec<ApiKey>> {
        let keys = key_entity::Entity::find()
            .all(&self.db)
            .await?;
        Ok(keys)
    }

    async fn update_status(&self, api_key: &str, status: KeyStatus) -> Result<()> {
        key_entity::Entity::update_many()
            .filter(key_entity::Column::Key.eq(api_key))
            .set(key_entity::ActiveModel {
                status: Set(serde_json::to_string(&status)
                    .unwrap_or_else(|_| "inactive".to_string())),
                ..Default::default()
            })
            .exec(&self.db)
            .await?;
        Ok(())
    }
}

#[async_trait]
impl ApiKeyStatsStorage for SqliteKeyStorage {
    async fn get_stats(&self, api_key: &str) -> Result<Option<ApiKeyStats>> {
        let stats = key_stats_entity::Entity::find()
            .filter(key_stats_entity::Column::Key.eq(api_key))
            .one(&self.db)
            .await?;
        
        Ok(stats.map(|s| s.into()))
    }

    async fn update_stats(&self, api_key: &str, stats: ApiKeyStats) -> Result<()> {
        let active_model: key_stats_entity::ActiveModel = key_stats_entity::ActiveModel {
            key: Set(api_key.to_string()),
            requests_today: Set(stats.requests_today as i64),
            requests_this_hour: Set(0),
            requests_this_minute: Set(0),
            last_request_at: Set(Some(stats.last_used_at)),
            total_requests: Set(stats.total_requests as i64),
            total_tokens: Set(0),
            total_audio_seconds: Set(0),
            updated_at: Set(chrono::Utc::now()),
        };
        
        key_stats_entity::Entity::insert(active_model)
            .on_conflict(
                OnConflict::column(key_stats_entity::Column::Key)
                    .update_columns([
                        key_stats_entity::Column::RequestsToday,
                        key_stats_entity::Column::RequestsThisHour,
                        key_stats_entity::Column::RequestsThisMinute,
                        key_stats_entity::Column::LastRequestAt,
                        key_stats_entity::Column::TotalRequests,
                        key_stats_entity::Column::TotalTokens,
                        key_stats_entity::Column::TotalAudioSeconds,
                        key_stats_entity::Column::UpdatedAt,
                    ])
                    .to_owned()
            )
            .exec(&self.db)
            .await?;
            
        Ok(())
    }
}
