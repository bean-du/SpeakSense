use chrono::{DateTime, Utc};
use serde::{Serialize, Deserialize};

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct ApiKeyInfo {
    pub key: String,
    pub name: String,
    pub created_at: DateTime<Utc>,
    pub expires_at: Option<DateTime<Utc>>,
    pub permissions: Vec<Permission>,
    pub rate_limit: RateLimit,
    pub status: KeyStatus,
}



#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum Permission {
    Transcribe,
    SpeakerDiarization,
    EmotionRecognition,
    Admin,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct RateLimit {
    pub requests_per_minute: u32,
    pub requests_per_hour: u32,
    pub requests_per_day: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum KeyStatus {
    Active,
    Suspended,
    Expired,
} 

impl Default for KeyStatus {
    fn default() -> Self {
        KeyStatus::Expired
    }
}

