//! Shared server state and configuration.

use std::sync::Arc;
use std::time::Instant;

use nimbus_edge_core::backend::ModelInfo;
use nimbus_edge_core::scheduler::Scheduler;

#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// The model id reported by the API (`model` echo, /v1/models).
    pub served_model_name: String,
    /// When set, /v1/* requires `Authorization: Bearer <token>`.
    /// /healthz always stays open for liveness probes.
    pub auth_token: Option<String>,
    /// When set (Some), emit CORS headers for the listed origins ("*" allowed).
    /// None = CORS off entirely (default; see SECURITY.md).
    pub cors_origins: Option<Vec<String>>,
}

pub struct AppState {
    pub scheduler: Arc<Scheduler>,
    pub model: ModelInfo,
    pub cfg: ServerConfig,
    pub started: Instant,
}

pub type SharedState = Arc<AppState>;
