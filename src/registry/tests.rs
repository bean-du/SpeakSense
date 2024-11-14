use super::*;
use super::etcd::EtcdRegistry;
use tokio::time::sleep;
use std::time::Duration;
use std::collections::HashMap;
use tokio::task;

async fn setup_test_registry() -> EtcdRegistry {
    // Use local etcd server for testing
    let endpoints = vec!["http://localhost:2379".to_string()];
    let lease_ttl = 10; // 10 seconds lease time
    EtcdRegistry::new(endpoints, lease_ttl).await.expect("Failed to create registry")
}

fn create_test_service_info(name: &str) -> ServiceInfo {
    let node = Node {
        id: format!("{}-test-node", name),
        address: "127.0.0.1:8080".to_string(),
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

    ServiceInfo {
        name: name.to_string(),
        version: "0.1.0".to_string(),
        protocol: "grpc".to_string(),
        metadata: None,
        endpoints: vec![],
        nodes: vec![node],
    }
}

#[tokio::test]
async fn test_registry_basic_lifecycle() {
    let registry = setup_test_registry().await;
    let service_info = create_test_service_info("test-service");
    let node_id = service_info.nodes[0].id.clone();

    // Test registering a service
    registry.register(service_info.clone()).await.expect("Failed to register service");

    // Start heartbeat
    registry.start_heartbeat().await.expect("Failed to start heartbeat");

    // Wait for a while to ensure heartbeat works
    sleep(Duration::from_secs(5)).await;

    // Stop heartbeat
    registry.stop_heartbeat().await.expect("Failed to stop heartbeat");

    // Deregister service
    registry.deregister(&service_info.name, &node_id).await.expect("Failed to deregister service");
}

#[tokio::test]
async fn test_registry_lease_renewal() {
    let registry = setup_test_registry().await;
    let service_info = create_test_service_info("test-service-lease");

    // Register service
    registry.register(service_info.clone()).await.expect("Failed to register service");

    // Start heartbeat
    registry.start_heartbeat().await.expect("Failed to start heartbeat");

    // Wait for a while longer than the lease time
    sleep(Duration::from_secs(15)).await;

    // The service should still exist because the heartbeat will renew the lease
    let mut client = etcd_client::Client::connect(["http://localhost:2379"], None).await.unwrap();
    let key = format!("/micro/registry/{}/{}-test-node", service_info.name, service_info.name);
    let response = client.get(key.clone(), None).await.expect("Failed to get key from etcd");
    assert!(response.count() > 0, "Service registration should still exist");

    // Stop heartbeat
    registry.stop_heartbeat().await.expect("Failed to stop heartbeat");

    // Wait for the lease to expire
    sleep(Duration::from_secs(15)).await;

    // Now the service should be automatically removed
    let response = client.get(key, None).await.expect("Failed to get key from etcd");
    assert_eq!(response.count(), 0, "Service registration should be removed after lease expiration");
}

#[tokio::test]
async fn test_registry_concurrent_services() {
    let registry = setup_test_registry().await;
    let mut handles = vec![];

    // Register multiple services concurrently
    for i in 0..5 {
        let registry = registry.clone();
        let service_info = create_test_service_info(&format!("test-service-{}", i));
        
        let handle = task::spawn(async move {
            registry.register(service_info.clone()).await.expect("Failed to register service");
            registry.start_heartbeat().await.expect("Failed to start heartbeat");
            service_info
        });
        
        handles.push(handle);
    }

    // Wait for all services to register
    let mut services = Vec::new();
    for handle in handles {
        services.push(handle.await.unwrap());
    }
    
    // Wait for a while to ensure heartbeat works
    sleep(Duration::from_secs(5)).await;

    // Deregister services concurrently
    let mut deregister_handles = vec![];
    for service in services {
        let registry = registry.clone();
        let node_id = service.nodes[0].id.clone();
        
        let handle = task::spawn(async move {
            registry.stop_heartbeat().await.expect("Failed to stop heartbeat");
            registry.deregister(&service.name, &node_id).await.expect("Failed to deregister service");
        });
        
        deregister_handles.push(handle);
    }

    // Wait for all services to deregister
    for handle in deregister_handles {
        handle.await.unwrap();
    }
}