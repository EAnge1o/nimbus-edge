//! Typed errors for the whole edge stack. The server crate maps these onto
//! OpenAI-style error JSON with matching HTTP status codes.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum EdgeError {
    /// 400 — malformed or invalid request parameter.
    #[error("{message}")]
    InvalidRequest {
        message: String,
        param: Option<String>,
        code: &'static str,
    },
    /// 400 — the parameter exists in the wider OpenAI API but is outside the
    /// documented compatible subset.
    #[error("{message}")]
    Unsupported {
        message: String,
        param: Option<String>,
        code: &'static str,
    },
    /// 400 — the served model has no chat template, so /v1/chat/completions
    /// would have to invent one.
    #[error(
        "model '{0}' has no chat template; use /v1/completions with a raw prompt, or import \
         the model with `nimbus-edge models import --chat-template chatml`"
    )]
    NoChatTemplate(String),
    /// 404 — unknown model or resource.
    #[error("{0}")]
    NotFound(String),
    /// 429 — the bounded queue is full.
    #[error("server is at capacity: {0}")]
    Busy(String),
    /// 500 — the inference backend failed.
    #[error("{0}")]
    Engine(String),
    /// 500 — unexpected internal failure.
    #[error("{0}")]
    Internal(String),
}

impl EdgeError {
    pub fn invalid(message: impl Into<String>) -> EdgeError {
        EdgeError::InvalidRequest {
            message: message.into(),
            param: None,
            code: "invalid_value",
        }
    }

    pub fn invalid_param(message: impl Into<String>, param: &str) -> EdgeError {
        EdgeError::InvalidRequest {
            message: message.into(),
            param: Some(param.to_string()),
            code: "invalid_value",
        }
    }

    pub fn unknown_parameter(name: &str, expected: &str) -> EdgeError {
        EdgeError::InvalidRequest {
            message: format!(
                "unknown request field {name:?}; this server implements a documented \
                 OpenAI-compatible subset and rejects fields it does not implement. \
                 Supported fields: {expected}"
            ),
            param: Some(name.to_string()),
            code: "unknown_parameter",
        }
    }

    pub fn unsupported(message: impl Into<String>, param: &str) -> EdgeError {
        EdgeError::Unsupported {
            message: message.into(),
            param: Some(param.to_string()),
            code: "unsupported_parameter",
        }
    }

    pub fn not_found(message: impl Into<String>) -> EdgeError {
        EdgeError::NotFound(message.into())
    }

    pub fn engine(message: impl Into<String>) -> EdgeError {
        EdgeError::Engine(message.into())
    }

    /// Machine-readable error code, used as the `code` field of the OpenAI
    /// error object.
    pub fn code(&self) -> &'static str {
        match self {
            EdgeError::InvalidRequest { code, .. } => code,
            EdgeError::Unsupported { code, .. } => code,
            EdgeError::NoChatTemplate(_) => "no_chat_template",
            EdgeError::NotFound(_) => "not_found",
            EdgeError::Busy(_) => "queue_full",
            EdgeError::Engine(_) => "generation_failed",
            EdgeError::Internal(_) => "internal_error",
        }
    }
}
