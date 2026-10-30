//! Opt-in CORS for browser-based clients (Open WebUI, LibreChat, ...).
//!
//! Default is **off**: no CORS headers are emitted, which keeps the
//! same-origin posture of a localhost dev server (see SECURITY.md). Enabling
//! it — `--cors-origins "*"` or an explicit origin list — lets browser apps
//! on other origins call the API directly, the way Ollama users expect.

use axum::extract::{Request, State};
use axum::http::{header, HeaderValue, Method, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};

use crate::state::SharedState;

pub async fn cors(State(state): State<SharedState>, req: Request, next: Next) -> Response {
    let Some(config) = state.cfg.cors_origins.clone() else {
        return next.run(req).await;
    };

    let origin = req
        .headers()
        .get(header::ORIGIN)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());

    // Preflight: answer directly, no route is invoked.
    if req.method() == Method::OPTIONS {
        let mut resp = StatusCode::NO_CONTENT.into_response();
        apply(&mut resp, &config, origin.as_deref(), true);
        return resp;
    }

    let mut resp = next.run(req).await;
    apply(&mut resp, &config, origin.as_deref(), false);
    resp
}

fn apply(resp: &mut Response, config: &[String], origin: Option<&str>, preflight: bool) {
    let headers = resp.headers_mut();
    let allowed = match origin {
        Some(o) if config.iter().any(|c| c == "*" || c == o) => Some(o.to_string()),
        // Non-browser clients (no Origin header) are unaffected by CORS.
        None if config.iter().any(|c| c == "*") => Some("*".to_string()),
        _ => None,
    };
    let Some(allowed) = allowed else { return };
    if let Ok(v) = HeaderValue::from_str(&allowed) {
        headers.insert(header::ACCESS_CONTROL_ALLOW_ORIGIN, v);
    }
    headers.insert(header::VARY, HeaderValue::from_static("Origin"));
    if preflight {
        headers.insert(
            header::ACCESS_CONTROL_ALLOW_METHODS,
            HeaderValue::from_static("GET, POST, OPTIONS"),
        );
        headers.insert(
            header::ACCESS_CONTROL_ALLOW_HEADERS,
            HeaderValue::from_static("Content-Type, Authorization"),
        );
        headers.insert(
            header::ACCESS_CONTROL_MAX_AGE,
            HeaderValue::from_static("86400"),
        );
    }
}
