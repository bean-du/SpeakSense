use sea_orm::entity::prelude::*;
use chrono::{DateTime, Utc};

#[derive(Clone, Debug, PartialEq, DeriveEntityModel)]
#[sea_orm(table_name = "api_key_stats")]
pub struct Model {
    #[sea_orm(primary_key)]
    pub key: String,
    pub requests_today: i64,
    pub requests_this_hour: i64,
    pub requests_this_minute: i64,
    pub last_request_at: Option<DateTime<Utc>>,
    pub total_requests: i64,
    pub total_tokens: i64,
    pub total_audio_seconds: i64,
    pub updated_at: DateTime<Utc>,
}

#[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
pub enum Relation {}

impl ActiveModelBehavior for ActiveModel {}
