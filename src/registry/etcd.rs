use super::{Registry, ServiceInfo};
use async_trait::async_trait;
use etcd_client::{Client, PutOptions};
use tokio::sync::mpsc;
use tokio::time::{interval, Duration};
use anyhow::Result;
use tracing::{info, error};
use std::sync::Arc;
use tokio::sync::Mutex;

#[derive(Clone)]
pub struct EtcdRegistry {
    client: Arc<Mutex<Client>>,
    lease_id: Arc<Mutex<Option<i64>>>,
    lease_ttl: i64,
    heartbeat_tx: Arc<Mutex<Option<mpsc::Sender<()>>>>,
}

impl EtcdRegistry {
    pub async fn new(endpoints: Vec<String>, lease_ttl: i64) -> Result<Self> {
        let client = Client::connect(endpoints, None).await?;
        
        Ok(Self {
            client: Arc::new(Mutex::new(client)),
            lease_id: Arc::new(Mutex::new(None)),
            lease_ttl,
            heartbeat_tx: Arc::new(Mutex::new(None)),
        })
    }

    async fn create_lease(&self) -> Result<i64> {
        let  client = self.client.lock().await;
        let mut lease = client.lease_client();
        let lease_resp = lease.grant(self.lease_ttl, None).await?;
        Ok(lease_resp.id())
    }

    async fn revoke_lease(&self, lease_id: i64) -> Result<()> {
        let  client = self.client.lock().await;
        let mut lease = client.lease_client();
        lease.revoke(lease_id).await?;
        Ok(())
    }

    fn get_service_key(&self, service_name: &str, node_id: &str) -> String {
        format!("/micro/registry/{}/{}", service_name, node_id)
    }
}

#[async_trait]
impl Registry for EtcdRegistry {
    async fn register(&self, service_info: ServiceInfo) -> Result<()> {
        // Create lease
        let lease_id = self.create_lease().await?;
        *self.lease_id.lock().await = Some(lease_id);

        // Put service info with lease
        let key = self.get_service_key(&service_info.name, &service_info.nodes[0].id);
        let value = serde_json::to_string(&service_info)?;
        
        let options = PutOptions::new().with_lease(lease_id);
        let mut client = self.client.lock().await;
        client.put(key, value, Some(options)).await?;
        info!("Registered service to etcd: {}", service_info.name);

        Ok(())
    }

    async fn deregister(&self, service_name: &str, node_id: &str) -> Result<()> {
        // Delete service info
        let key = self.get_service_key(service_name, node_id);
        let mut client = self.client.lock().await;
        client.delete(key, None).await?;

        // Revoke lease if exists
        let lease_id = {
            let mut lease_guard = self.lease_id.lock().await;
            lease_guard.take()
        };

        if let Some(lease_id) = lease_id {
            self.revoke_lease(lease_id).await?;
        }

        info!("Deregistered service from etcd: {}", service_name);
        Ok(())
    }

    async fn start_heartbeat(&self) -> Result<()> {
        let lease_id = {
            let lease_guard = self.lease_id.lock().await;
            lease_guard.ok_or_else(|| anyhow::anyhow!("No lease found"))?
        };

        let (tx, mut rx) = mpsc::channel(1);
        *self.heartbeat_tx.lock().await = Some(tx);

        let client = Arc::clone(&self.client);
        let lease_id_copy = lease_id;

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(3));
            
            loop {
                tokio::select! {
                    _ = interval.tick() => {
                        let  client = client.lock().await;
                        let mut lease = client.lease_client();
                        if let Err(e) = lease.keep_alive(lease_id_copy).await {
                            error!("Failed to keep lease alive: {}", e);
                        }
                    }
                    _ = rx.recv() => {
                        info!("Stopping heartbeat");
                        break;
                    }
                }
            }
        });

        Ok(())
    }

    async fn stop_heartbeat(&self) -> Result<()> {
        if let Some(tx) = self.heartbeat_tx.lock().await.take() {
            let _ = tx.send(()).await;
        }
        Ok(())
    }
}
