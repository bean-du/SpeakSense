use super::*;
use crate::auth::types::{Permission, RateLimit};
use chrono::{Duration, Utc};
use tempfile::NamedTempFile;

#[allow(dead_code)]
async fn setup_test_db() -> (SqliteKeyStorage, NamedTempFile) {
    let temp_file = NamedTempFile::new().unwrap();
    let db_url = format!("sqlite:{}?mode=rwc", temp_file.path().display());
    let storage = SqliteKeyStorage::new(&db_url).await.unwrap();
    (storage, temp_file)
}

#[allow(dead_code)]
fn create_test_api_key(key: &str) -> ApiKey {
    let mut api_key = ApiKey {
        key: key.to_string(),
        name: "Test Key".to_string(),
        created_at: Utc::now(),
        expires_at: Some(Utc::now() + Duration::days(30)),
        permissions: String::new(),
        rate_limit: String::new(),
        status: "active".to_string(),
    };

    let permissions = vec![Permission::Transcribe, Permission::SpeakerDiarization];
    let rate_limit = RateLimit {
        requests_per_minute: 60,
        requests_per_hour: 1000,
        requests_per_day: 10000,
    };

    api_key.set_permissions(permissions);
    api_key.set_rate_limit(rate_limit);
    api_key
}

#[tokio::test]
async fn test_create_and_get_key() {
    let (storage, _temp_file) = setup_test_db().await;
    let test_key = "test_key_1";
    let api_key = create_test_api_key(test_key);

    // Test create
    storage.create(test_key.to_string(), api_key.clone()).await.unwrap();

    // Test get
    let retrieved = storage.get(test_key).await.unwrap().unwrap();
    assert_eq!(retrieved.key, test_key);
    assert_eq!(retrieved.name, "Test Key");
    assert_eq!(retrieved.get_permissions(), vec![Permission::Transcribe, Permission::SpeakerDiarization]);
    assert_eq!(retrieved.get_rate_limit().requests_per_minute, 60);
    assert_eq!(retrieved.get_rate_limit().requests_per_hour, 1000);
    assert_eq!(retrieved.get_rate_limit().requests_per_day, 10000);
}

#[tokio::test]
async fn test_list_keys() {
    let (storage, _temp_file) = setup_test_db().await;
    
    // Create multiple keys
    for i in 1..=3 {
        let key = format!("test_key_{}", i);
        let api_key = create_test_api_key(&key);
        storage.create(key, api_key).await.unwrap();
    }

    // Test list
    let keys = storage.list().await.unwrap();
    assert_eq!(keys.len(), 3);
}

#[tokio::test]
async fn test_remove_key() {
    let (storage, _temp_file) = setup_test_db().await;
    let test_key = "test_key_remove";
    let api_key = create_test_api_key(test_key);

    // Create and verify key exists
    storage.create(test_key.to_string(), api_key).await.unwrap();
    assert!(storage.get(test_key).await.unwrap().is_some());

    // Remove key
    storage.remove(test_key).await.unwrap();
    assert!(storage.get(test_key).await.unwrap().is_none());
}

#[tokio::test]
async fn test_update_status() {
    let (storage, _temp_file) = setup_test_db().await;
    let test_key = "test_key_status";
    let api_key = create_test_api_key(test_key);

    // Create key
    storage.create(test_key.to_string(), api_key).await.unwrap();

    // Update status
    storage.update_status(test_key, KeyStatus::Expired).await.unwrap();

    // Verify status update
    let updated = storage.get(test_key).await.unwrap().unwrap();
    let status: KeyStatus = serde_json::from_str(&updated.status).unwrap();
    assert_eq!(status, KeyStatus::Expired);
}

#[tokio::test]
async fn test_get_nonexistent_key() {
    let (storage, _temp_file) = setup_test_db().await;
    let result = storage.get("nonexistent_key").await.unwrap();
    assert!(result.is_none());
} 