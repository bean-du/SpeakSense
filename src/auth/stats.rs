use chrono::{DateTime, Utc, Duration};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use super::types::ApiKeyInfo;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ApiKeyStats {
    pub total_requests: u64,
    pub requests_today: u64,
    pub last_used_at: DateTime<Utc>,
    pub requests_per_day: HashMap<String, u64>,
}

#[derive(Debug, Serialize, Deserialize, Default)]
pub struct ApiKeyUsageReport {
    pub key_info: ApiKeyInfo,
    pub stats: ApiKeyStats,
    pub usage_summary: UsageSummary,
}

#[derive(Debug, Serialize, Deserialize, Default)]
pub struct UsageSummary {
    pub average_daily_requests: f64,
    pub peak_daily_requests: u64,
    pub days_until_expiry: i64,
}

impl ApiKeyStats {
    pub fn new() -> Self {
        Self {
            total_requests: 0,
            requests_today: 0,
            last_used_at: Utc::now(),
            requests_per_day: HashMap::new(),
        }
    }

    pub fn update(&mut self) {
        let today = Utc::now().date_naive().to_string();
        self.total_requests += 1;
        self.last_used_at = Utc::now();
        
        let today_requests = self.requests_per_day.entry(today.clone()).or_insert(0);
        *today_requests += 1;
        self.requests_today = *today_requests;

        let thirty_days_ago = (Utc::now() - Duration::days(30)).date_naive().to_string();
        self.requests_per_day.retain(|date, _| date >= &thirty_days_ago);
    }
} 