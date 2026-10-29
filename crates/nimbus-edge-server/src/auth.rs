//! Optional bearer-token auth for /v1/* routes. `/healthz` stays open so
//! liveness probes work without credentials (documented in SECURITY.md).

use axum::extract::{Request, State};
use axum::http::StatusCode;
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use axum::Json;

use nimbus_edge_core::error::EdgeError;
use nimbus_edge_core::openai::ErrorBody;

use crate::state::SharedState;

pub async fn require_bearer(
    State(state): State<SharedState>,
    req: Request,
    next: Next,
) -> Response {
    let Some(expected) = state.cfg.auth_token.clone() else {
        return next.run(req).await;
    };
    let provided = req
        .headers()
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .unwrap_or_default();
    if token_matches(provided, &expected) {
        return next.run(req).await;
    }
    (
        StatusCode::UNAUTHORIZED,
        Json(ErrorBody::from_error(&EdgeError::InvalidRequest {
            message: "missing or invalid API key; send 'Authorization: Bearer <token>'".to_string(),
            param: None,
            code: "invalid_api_key",
        })),
    )
        .into_response()
}

/// Constant-time comparison so a wrong token cannot be probed byte by byte
/// via timing. Defense in depth for a localhost dev server.
fn token_matches(a: &str, b: &str) -> bool {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    if a.len() != b.len() {
        return false;
    }
    a.iter()
        .zip(b.iter())
        .fold(0u8, |acc, (x, y)| acc | (x ^ y))
        == 0
}
