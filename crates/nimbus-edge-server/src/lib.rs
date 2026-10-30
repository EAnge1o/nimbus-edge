//! nimbus-edge-server: an HTTP/SSE server exposing the documented
//! OpenAI-compatible subset on top of `nimbus-edge-core`'s scheduler.
//!
//! Layering: this crate knows HTTP, SSE and error mapping; it never touches
//! the inference engine directly — everything goes through the scheduler's
//! event stream, which is also what makes client-disconnect cancellation
//! uniform for both streaming and non-streaming requests.

use std::sync::Arc;
use std::time::Instant;

use axum::Router;
use tokio::net::TcpListener;

use nimbus_edge_core::backend::ModelInfo;
use nimbus_edge_core::scheduler::Scheduler;

pub mod auth;
pub mod cors;
pub mod errors;
pub mod routes;
pub mod sse;
pub mod state;

pub use state::{AppState, ServerConfig};

/// Build the application router (health, models, chat/completions).
pub fn build_router(scheduler: Arc<Scheduler>, model: ModelInfo, cfg: ServerConfig) -> Router {
    let state = Arc::new(AppState {
        scheduler,
        model,
        cfg,
        started: Instant::now(),
    });
    routes::router(state.clone()).layer(axum::middleware::from_fn_with_state(state, cors::cors))
}

/// Serve on an already-bound listener until the shutdown signal fires, then
/// cancel all in-flight and queued requests.
pub async fn serve(
    listener: TcpListener,
    scheduler: Arc<Scheduler>,
    model: ModelInfo,
    cfg: ServerConfig,
) -> std::io::Result<()> {
    let app = build_router(scheduler.clone(), model, cfg);
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    scheduler.shutdown();
    Ok(())
}

/// Ctrl-C (all platforms) or SIGTERM (unix).
async fn shutdown_signal() {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{signal, SignalKind};
        let mut term = signal(SignalKind::terminate()).expect("install SIGTERM handler");
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {}
            _ = term.recv() => {}
        }
    }
    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
    }
}
