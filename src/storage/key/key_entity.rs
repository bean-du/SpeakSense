use sea_orm::entity::prelude::*;
use chrono::{DateTime, Utc};
use crate::auth::types::{Permission, RateLimit};

#[derive(Clone, Debug, PartialEq, DeriveEntityModel)]
#[sea_orm(table_name = "api_keys")]
pub struct Model {
    #[sea_orm(primary_key)]
    pub key: String,
    pub name: String,
    pub created_at: DateTime<Utc>,
    pub expires_at: Option<DateTime<Utc>>,
    #[sea_orm(column_type = "Text")]
    pub permissions: String,
    #[sea_orm(column_type = "Text")]
    pub rate_limit: String,
    pub status: String,
}

#[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
pub enum Relation {}

impl ActiveModelBehavior for ActiveModel {}

impl Model {
    pub fn get_permissions(&self) -> Vec<Permission> {
        serde_json::from_str(&self.permissions).unwrap_or_default()
    }

    pub fn set_permissions(&mut self, permissions: Vec<Permission>) {
        self.permissions = serde_json::to_string(&permissions).unwrap_or_default();
    }

    pub fn get_rate_limit(&self) -> RateLimit {
        serde_json::from_str(&self.rate_limit).unwrap_or_default()
    }

    pub fn set_rate_limit(&mut self, rate_limit: RateLimit) {
        self.rate_limit = serde_json::to_string(&rate_limit).unwrap_or_default();
    }
} 