pub mod etcd;

#[cfg(test)]
mod tests;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use anyhow::Result;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    pub id: String,
    pub address: String,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceInfo {
    pub name: String,
    pub version: String,
    pub protocol: String,
    pub metadata: Option<HashMap<String, String>>,
    pub endpoints: Vec<String>,
    pub nodes: Vec<Node>,
}

#[async_trait]
pub trait Registry: Send + Sync {
    /// Register service to registry
    async fn register(&self, service_info: ServiceInfo) -> Result<()>;
    
    /// Deregister service from registry
    async fn deregister(&self, service_name: &str, node_id: &str) -> Result<()>;
    
    /// Start the registry heartbeat
    async fn start_heartbeat(&self) -> Result<()>;
    
    /// Stop the registry heartbeat
    async fn stop_heartbeat(&self) -> Result<()>;
}