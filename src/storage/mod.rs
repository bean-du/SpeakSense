pub mod task;
pub mod key;

use sea_orm::{Database, DatabaseConnection};
use anyhow::Result;

pub type Db = DatabaseConnection;

pub async fn init_db(database_url: &str) -> Result<Db> {
    let db = Database::connect(database_url).await?;
    Ok(db)
}

// 重导出常用类型
pub use task::{TaskStorage, sqlite::SqliteTaskStorage};
