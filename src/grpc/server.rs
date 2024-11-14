use crate::registry::{Registry, ServiceInfo, Node};
use std::collections::HashMap;
use std::net::SocketAddr;
use tonic::transport::Server;
use uuid::Uuid;
use local_ip_address::local_ip;
use std::sync::Arc;
use anyhow::Result;
use tonic::codegen::Service;
use tonic::body::BoxBody;
use http::{Request, Response};
use tonic::server::NamedService;
use std::convert::Infallible;
use tracing::{info, error};
pub struct GrpcServer {
    registry: Arc<dyn Registry>,
    service_name: String,
    service_version: String,
    server_id: String,
}

impl GrpcServer {
    pub fn new(registry: Arc<dyn Registry>, service_name: String, service_version: String) -> Self {
        Self {
            registry,
            service_name,
            service_version,
            server_id: Uuid::new_v4().to_string(),
        }
    }

    pub async fn start<S>(&self, service: S, port: u16) -> Result<()> 
    where
    S: Service<Request<BoxBody>, Response = Response<BoxBody>, Error = Infallible>
            + NamedService
            + Clone
            + Send
            + Sync
            + 'static,
        S::Future: Send + 'static,
    {
        // Get local IP
        let ip = local_ip()?;
        let addr = SocketAddr::new("0.0.0.0".parse().unwrap(), port);
        
        let registry_addr = format!("{}:{}", ip, port);
        // Create service info
        let node = Node {
            id: format!("{}-{}-{}", self.service_name, "grpc", self.server_id),
            address: registry_addr.to_string(),
            metadata: {
                let mut map = HashMap::new();
                map.insert("broker".to_string(), "http".to_string());
                map.insert("protocol".to_string(), "grpc".to_string());
                map.insert("registry".to_string(), "etcd".to_string());
                map.insert("server".to_string(), "grpc".to_string());
                map.insert("transport".to_string(), "grpc".to_string());
                map
            },
        };

        let service_info = ServiceInfo {
            name: self.service_name.clone(),
            version: self.service_version.clone(),
            protocol: "grpc".to_string(),
            metadata: None,
            endpoints: vec![],
            nodes: vec![node],
        };

        // Register service
        match self.registry.register(service_info).await {
            Ok(_) => info!("Service registered successfully"),
            Err(e) => error!("Failed to register service: {}", e),
        }
        
        // Start heartbeat
        match self.registry.start_heartbeat().await {
            Ok(_) => info!("Heartbeat started successfully"),
            Err(e) => error!("Failed to start heartbeat: {}", e),
        }

        // Start gRPC server
        Server::builder()
            .add_service(service)
            .serve(addr)
            .await?;

        Ok(())
    }

    pub async fn stop(&self) -> Result<()> {
        self.registry.stop_heartbeat().await?;
        self.registry.deregister(
            &self.service_name,
            &format!("{}-{}", self.service_name, self.server_id)
        ).await?;
        Ok(())
    }
} 


#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::etcd::EtcdRegistry;
    use crate::grpc::handlers::asr::AsrService;
    use crate::asr::whisper::WhisperAsr;
    use crate::grpc::pb::asr::asr_server;
    #[tokio::test]
    async fn test_grpc_server() -> Result<()> {
        let registry = Arc::new(EtcdRegistry::new(vec!["http://localhost:2379".to_string()], 10).await.unwrap());
        let server = GrpcServer::new(registry, "asr".to_string(), "0.1.0".to_string());
        let asr = Arc::new(WhisperAsr::new("./models/ggml-large-v3.bin".to_string())?);
        let asr_service = AsrService::new(asr);

        server.start(asr_server::AsrServer::new(asr_service), 7300).await?;
        Ok(())
    }
}