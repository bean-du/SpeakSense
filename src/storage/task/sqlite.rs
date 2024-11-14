use async_trait::async_trait;
use anyhow::Result;
use chrono::{DateTime, Utc};
use sea_orm::{
    DatabaseConnection, EntityTrait, QueryFilter, ColumnTrait, QueryOrder,
    QuerySelect, Condition, ConnectionTrait, DbBackend, Statement,
    ActiveModelTrait, Set, IntoActiveModel,
};
use crate::web::Pagination;
use tracing::info;
use crate::schedule::types::TaskStatus;
use sea_query;

use super::TaskStorage;
use super::entity::{self, Model as TaskModel};
use sea_orm::{ConnectOptions, Database};

pub struct SqliteTaskStorage {
    db: DatabaseConnection,
}

impl SqliteTaskStorage {
    pub async fn new(database_url: &str) -> Result<Self> {
        info!("Initializing SQLite task storage at {}", database_url);

        let db = Database::connect(
            ConnectOptions::new(database_url.to_owned())
                .sqlx_logging(false)
                .to_owned()
        ).await?;
        
        db.execute(Statement::from_string(
            DbBackend::Sqlite,
            r#"
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY NOT NULL,
                status TEXT NOT NULL,
                config TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT,
                result TEXT,
                error TEXT,
                priority INTEGER NOT NULL,
                retry_count INTEGER NOT NULL,
                max_retries INTEGER NOT NULL,
                timeout INTEGER
            )
            "#.to_owned(),
        ))
        .await?;

        Ok(Self { db })
    }
}

#[async_trait]
impl TaskStorage for SqliteTaskStorage {
    async fn create(&self, model: &TaskModel) -> Result<()> {
        let active_model = model.clone().into_active_model();
        entity::Entity::insert(active_model)
            .on_conflict(
                sea_query::OnConflict::column(entity::Column::Id)
                    .update_columns([
                        entity::Column::UpdatedAt,
                        entity::Column::Status,
                        entity::Column::StartedAt,
                        entity::Column::CompletedAt,
                        entity::Column::Result,
                        entity::Column::Error,
                    ])
                    .to_owned()
            )
            .exec(&self.db)
            .await?;
        Ok(())
    }
    
    async fn list(&self, pagination: &Pagination) -> Result<Vec<TaskModel>> {
        let pagination = pagination.check();

        let models = entity::Entity::find()
            .order_by_asc(entity::Column::CreatedAt)
            .limit(pagination.limit())
            .offset(pagination.offset())
            .all(&self.db)
            .await?;
        Ok(models)
    }

    async fn get_pending_by_priority(&self, limit: usize) -> Result<Vec<TaskModel>> {
        let pending_status = serde_json::to_string(&TaskStatus::Pending)?;
        let models = entity::Entity::find()
            .filter(entity::Column::Status.eq(pending_status))
            .order_by_asc(entity::Column::Priority)
            .order_by_asc(entity::Column::CreatedAt)
            .limit(limit as u64)
            .all(&self.db)
            .await?;
        Ok(models)
    }

    async fn get(&self, task_id: &str) -> Result<Option<TaskModel>> {
        Ok(entity::Entity::find_by_id(task_id)
            .one(&self.db)
            .await?)
    }

    async fn update(&self, task_id: &str, status: &str) -> Result<()> {
        let now = Utc::now();
        if let Some(model) = entity::Entity::find_by_id(task_id).one(&self.db).await? {
            let mut active_model = model.into_active_model();
            active_model.status = Set(status.to_string());
            active_model.updated_at = Set(now);
            
            if status == serde_json::to_string(&TaskStatus::Processing)? {
                active_model.started_at = Set(Some(now));
            }
            if status == serde_json::to_string(&TaskStatus::Completed)? {
                active_model.completed_at = Set(Some(now));
            }
            
            active_model.update(&self.db).await?;
        }
        Ok(())
    }

    async fn delete(&self, task_id: &str) -> Result<()> {
        entity::Entity::delete_by_id(task_id)
            .exec(&self.db)
            .await?;
        Ok(())
    }

    async fn get_timeouted(&self) -> Result<Vec<TaskModel>> {
        let processing_status = serde_json::to_string(&TaskStatus::Processing)?;
        let now = Utc::now().timestamp();

        let statement = Statement::from_string(
            DbBackend::Sqlite,
            format!(
                r#"
                SELECT * FROM tasks 
                WHERE status = '{}'
                AND started_at IS NOT NULL 
                AND timeout IS NOT NULL
                AND (strftime('%s', started_at) + timeout) < {}
                "#,
                processing_status, now
            ),
        );

        let models = entity::Entity::find()
            .from_raw_sql(statement)
            .all(&self.db)
            .await?;
        
        Ok(models)
    }

    async fn cleanup_old(&self, before: DateTime<Utc>) -> Result<u64> {
        let condition = Condition::any()
            .add(entity::Column::Status.contains("Completed"))
            .add(entity::Column::Status.contains("Failed"));

        let result = entity::Entity::delete_many()
            .filter(condition)
            .filter(entity::Column::UpdatedAt.lt(before))
            .exec(&self.db)
            .await?;

        Ok(result.rows_affected)
    }

    async fn get_by_status(&self, status: &str) -> Result<Vec<TaskModel>> {
        let models = entity::Entity::find()
            .filter(entity::Column::Status.eq(status))
            .order_by_desc(entity::Column::Priority)
            .order_by_asc(entity::Column::CreatedAt)
            .all(&self.db)
            .await?;
        
        Ok(models)
    }
}

