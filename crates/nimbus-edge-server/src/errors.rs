//! Mapping [`EdgeError`] onto HTTP responses with OpenAI-style error JSON.

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;

use nimbus_edge_core::error::EdgeError;
use nimbus_edge_core::openai::ErrorBody;

pub struct ApiError(pub EdgeError);

impl From<EdgeError> for ApiError {
    fn from(e: EdgeError) -> ApiError {
        ApiError(e)
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let status =
            StatusCode::from_u16(self.0.http_status()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
        let mut response = (status, Json(ErrorBody::from_error(&self.0))).into_response();
        if matches!(self.0, EdgeError::Busy(_)) {
            response
                .headers_mut()
                .insert("Retry-After", axum::http::HeaderValue::from_static("1"));
        }
        response
    }
}

/// Map an axum JSON extractor rejection (malformed body, wrong types, unknown
/// fields) onto the OpenAI error shape. The rejection text from serde already
/// names the offending field; keep it visible.
pub fn rejection(rej: axum::extract::rejection::JsonRejection) -> ApiError {
    let message = match &rej {
        axum::extract::rejection::JsonRejection::JsonDataError(e) => format!(
            "invalid request body: {e}; this server implements a documented \
             OpenAI-compatible subset and rejects unknown or invalid fields"
        ),
        other => format!("invalid request body: {other}"),
    };
    ApiError(EdgeError::InvalidRequest {
        message,
        param: None,
        code: "invalid_request_body",
    })
}
