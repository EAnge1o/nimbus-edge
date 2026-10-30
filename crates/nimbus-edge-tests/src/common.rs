//! Shared helpers for integration tests: spin up an in-process server over
//! the real axum stack on an ephemeral port, backed by a random-weight tiny
//! checkpoint (no network, no large downloads).

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use nimbus_edge_core::backend::{Backend, NimbusBackend, PrefixCacheConfig};
use nimbus_edge_core::chat::ChatTemplate;
use nimbus_edge_core::registry::Registry;
use nimbus_edge_core::scheduler::{Scheduler, SchedulerConfig};
use nimbus_edge_server::{build_router, ServerConfig};

pub use nimbus_edge_core::fixture::write_random_checkpoint;

/// A unique temp directory per test (cleaned on drop, best effort).
pub struct TempDir(PathBuf);
impl TempDir {
    pub fn new(tag: &str) -> TempDir {
        let n = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("nb-{tag}-{n}"));
        std::fs::create_dir_all(&dir).unwrap();
        TempDir(dir)
    }
    pub fn path(&self) -> &Path {
        &self.0
    }
}
impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Import a checkpoint into a temp registry under `name`.
pub fn import(registry_root: &Path, source: &Path, name: &str, template: ChatTemplate) -> PathBuf {
    let registry = Registry::open(registry_root);
    registry
        .import(
            source,
            name,
            template,
            "fp32",
            false,
            "2026-01-01T00:00:00Z".to_string(),
        )
        .expect("import succeeds");
    registry_root.join(name)
}

pub struct TestServer {
    pub base: String,
    pub client: reqwest::Client,
    _bg: tokio::task::JoinHandle<()>,
}

pub struct ServerOptions {
    pub template: ChatTemplate,
    pub max_concurrency: usize,
    pub max_queue_size: usize,
    pub prefix_cache: bool,
    pub auth_token: Option<String>,
    pub cors_origins: Option<Vec<String>>,
}

impl Default for ServerOptions {
    fn default() -> Self {
        ServerOptions {
            template: ChatTemplate::None,
            max_concurrency: 1,
            max_queue_size: 16,
            prefix_cache: false,
            auth_token: None,
            cors_origins: None,
        }
    }
}

/// Start a server for a checkpoint directory on 127.0.0.1:0.
pub async fn spawn_server(model_dir: &Path, opts: ServerOptions) -> TestServer {
    let backend = NimbusBackend::load(
        model_dir,
        nimbus_infer_rs::engine::QuantMode::F32,
        "tiny".to_string(),
        opts.template,
        PrefixCacheConfig {
            enabled: opts.prefix_cache,
            ..Default::default()
        },
    )
    .expect("backend loads");
    let info = backend.info();
    let scheduler = Scheduler::spawn(
        Arc::new(backend) as Arc<dyn Backend>,
        SchedulerConfig {
            max_queue_size: opts.max_queue_size,
            max_concurrency: opts.max_concurrency,
        },
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let app = build_router(
        scheduler,
        info,
        ServerConfig {
            served_model_name: "tiny".to_string(),
            auth_token: opts.auth_token,
            cors_origins: opts.cors_origins,
        },
    );
    let bg = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    TestServer {
        base,
        client: reqwest::Client::new(),
        _bg: bg,
    }
}

impl TestServer {
    /// POST JSON, return (status, body-as-Value).
    pub async fn post(&self, path: &str, body: serde_json::Value) -> (u16, serde_json::Value) {
        let resp = self
            .client
            .post(format!("{}{path}", self.base))
            .json(&body)
            .send()
            .await
            .expect("request completes");
        let status = resp.status().as_u16();
        let text = resp.text().await.unwrap_or_default();
        (
            status,
            serde_json::from_str(&text).unwrap_or(serde_json::json!({"raw": text})),
        )
    }

    pub async fn get(&self, path: &str) -> (u16, serde_json::Value) {
        let resp = self
            .client
            .get(format!("{}{path}", self.base))
            .send()
            .await
            .unwrap();
        let status = resp.status().as_u16();
        let text = resp.text().await.unwrap_or_default();
        (
            status,
            serde_json::from_str(&text).unwrap_or(serde_json::json!({"raw": text})),
        )
    }

    /// POST and return the raw SSE byte stream for inspection.
    pub async fn post_stream(
        &self,
        path: &str,
        body: serde_json::Value,
    ) -> impl futures_util::Stream<Item = reqwest::Result<Bytes>> {
        self.client
            .post(format!("{}{path}", self.base))
            .json(&body)
            .send()
            .await
            .expect("streaming request completes")
            .bytes_stream()
    }

    /// Collect the whole SSE session as a Vec<String> of `data:` payloads.
    pub async fn collect_sse(&self, path: &str, body: serde_json::Value) -> Vec<String> {
        use futures_util::StreamExt;
        let stream = self.post_stream(path, body).await;
        let mut bytes = Vec::new();
        let mut pinned = Box::pin(stream);
        let deadline = tokio::time::Instant::now() + Duration::from_secs(30);
        while let Some(chunk) = pinned.next().await {
            bytes.extend_from_slice(&chunk.unwrap());
            if tokio::time::Instant::now() > deadline {
                panic!("sse collection exceeded 30s");
            }
            let text = String::from_utf8_lossy(&bytes);
            if text.contains("data: [DONE]") {
                break;
            }
        }
        String::from_utf8_lossy(&bytes)
            .lines()
            .filter_map(|l| l.strip_prefix("data: "))
            .map(|s| s.to_string())
            .collect()
    }

    /// Poll /healthz until `pred` holds or the deadline passes.
    pub async fn wait_until(
        &self,
        pred: impl Fn(&serde_json::Value) -> bool,
        secs: u64,
    ) -> serde_json::Value {
        let deadline = tokio::time::Instant::now() + Duration::from_secs(secs);
        loop {
            let (_, body) = self.get("/healthz").await;
            if pred(&body) {
                return body;
            }
            if tokio::time::Instant::now() > deadline {
                panic!("healthz condition not reached in {secs}s; last: {body}");
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    }
}

/// The `bytes` type used by reqwest's stream items.
pub use bytes::Bytes;

/// A chat request body for the "tiny" model.
pub fn chat_body(content: &str, max_tokens: u32) -> serde_json::Value {
    serde_json::json!({
        "model": "tiny",
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
    })
}
