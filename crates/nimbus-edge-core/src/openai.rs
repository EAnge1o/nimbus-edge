//! OpenAI-compatible request/response types for the documented subset.
//!
//! Parsing is strict: unknown fields are rejected (`deny_unknown_fields`) so
//! clients never believe a parameter took effect when it did not. Parameters
//! that exist in the wider OpenAI API but are outside the subset are rejected
//! with [`EdgeError::Unsupported`], not silently ignored. The exact subset is
//! listed in the README ("API compatibility").

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::EdgeError;

pub const CHAT_FIELDS: &str =
    "model, messages, temperature, top_p, max_tokens, stop, seed, stream, user";
pub const COMPLETION_FIELDS: &str =
    "model, prompt, temperature, top_p, max_tokens, stop, seed, stream, user";

// ---------------------------------------------------------------------------
// Requests
// ---------------------------------------------------------------------------

/// The `stop` field accepts a string or an array of up to 4 strings.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum StopField {
    One(String),
    Many(Vec<String>),
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub top_p: Option<f64>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub stop: Option<StopField>,
    #[serde(default)]
    pub seed: Option<i64>,
    #[serde(default)]
    pub stream: bool,
    /// Caller identity metadata. Accepted and ignored: it has no effect on
    /// inference (documented in the README compatibility table).
    #[serde(default)]
    pub user: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompletionRequest {
    pub model: String,
    pub prompt: PromptField,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub top_p: Option<f64>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub stop: Option<StopField>,
    #[serde(default)]
    pub seed: Option<i64>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub user: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum PromptField {
    One(String),
    Many(Vec<String>),
}

impl PromptField {
    /// The subset accepts a single prompt string, or an array with exactly
    /// one element (some SDKs always send arrays).
    pub fn into_text(self) -> Result<String, EdgeError> {
        match self {
            PromptField::One(s) => Ok(s),
            PromptField::Many(v) if v.len() == 1 => Ok(v.into_iter().next().unwrap()),
            PromptField::Many(v) => Err(EdgeError::unsupported(
                format!(
                    "prompt arrays with {} elements are not supported; send a single prompt \
                     string (n>1 completions are outside the compatible subset)",
                    v.len()
                ),
                "prompt",
            )),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    /// String content only. Multimodal content arrays are rejected with a
    /// clear error instead of being flattened.
    pub content: Value,
}

impl ChatMessage {
    pub fn into_content(self) -> Result<(String, String), EdgeError> {
        let content = match self.content {
            Value::String(s) => s,
            Value::Array(_) => {
                return Err(EdgeError::invalid_param(
                    "message content must be a string; content arrays (multimodal messages) \
                     are outside the compatible subset",
                    "messages",
                ))
            }
            other => {
                return Err(EdgeError::invalid_param(
                    format!("message content must be a string, got {other}"),
                    "messages",
                ))
            }
        };
        match self.role.as_str() {
            "system" | "user" | "assistant" => Ok((self.role, content)),
            other => Err(EdgeError::invalid_param(
                format!(
                    "message role {other:?} is not supported; expected system, user or assistant"
                ),
                "messages",
            )),
        }
    }
}

// ---------------------------------------------------------------------------
// Validation -> engine parameters
// ---------------------------------------------------------------------------

/// Fully validated generation parameters, independent of HTTP.
#[derive(Debug, Clone)]
pub struct GenParams {
    pub max_new_tokens: u32,
    pub temperature: f32,
    pub top_p: f32,
    pub seed: Option<u64>,
    pub stop_strings: Vec<String>,
}

pub const MAX_TOKENS_CEILING: u32 = 16_384;

/// Shared validation for both endpoints. `fields` names the supported fields,
/// used in error messages.
pub fn validate_common(
    temperature: Option<f64>,
    top_p: Option<f64>,
    max_tokens: Option<u32>,
    stop: Option<StopField>,
    seed: Option<i64>,
) -> Result<GenParams, EdgeError> {
    if let Some(t) = temperature {
        if !(0.0..=2.0).contains(&t) || !t.is_finite() {
            return Err(EdgeError::invalid_param(
                format!("temperature must be a number in [0, 2], got {t}"),
                "temperature",
            ));
        }
    }
    if let Some(p) = top_p {
        if !(0.0..=1.0).contains(&p) || !p.is_finite() || p == 0.0 {
            return Err(EdgeError::invalid_param(
                format!("top_p must be a number in (0, 1], got {p}"),
                "top_p",
            ));
        }
    }
    if let Some(n) = max_tokens {
        if n == 0 {
            return Err(EdgeError::invalid_param(
                "max_tokens must be >= 1",
                "max_tokens",
            ));
        }
        if n > MAX_TOKENS_CEILING {
            return Err(EdgeError::invalid_param(
                format!("max_tokens must be <= {MAX_TOKENS_CEILING}"),
                "max_tokens",
            ));
        }
    }
    let stop_strings = match stop {
        None => Vec::new(),
        Some(StopField::One(s)) => vec![validate_stop_string(s)?],
        Some(StopField::Many(v)) => {
            if v.is_empty() {
                return Err(EdgeError::invalid_param(
                    "stop must contain at least one string",
                    "stop",
                ));
            }
            if v.len() > 4 {
                return Err(EdgeError::unsupported(
                    "stop accepts at most 4 sequences",
                    "stop",
                ));
            }
            v.into_iter()
                .map(validate_stop_string)
                .collect::<Result<Vec<_>, _>>()?
        }
    };
    let seed = match seed {
        None => None,
        Some(s) if s >= 0 => Some(s as u64),
        Some(s) => {
            return Err(EdgeError::invalid_param(
                format!("seed must be a non-negative integer, got {s}"),
                "seed",
            ))
        }
    };
    Ok(GenParams {
        max_new_tokens: max_tokens.unwrap_or(64),
        temperature: temperature.map(|t| t as f32).unwrap_or(1.0),
        top_p: top_p.map(|p| p as f32).unwrap_or(1.0),
        seed,
        stop_strings,
    })
}

fn validate_stop_string(s: String) -> Result<String, EdgeError> {
    if s.is_empty() {
        return Err(EdgeError::invalid_param(
            "stop sequences must be non-empty",
            "stop",
        ));
    }
    if s.chars().count() > 256 {
        return Err(EdgeError::invalid_param(
            "stop sequences must be at most 256 characters",
            "stop",
        ));
    }
    Ok(s)
}

/// Rejects parameter names the subset does not implement. Used by the server
/// for a clear, non-silent answer to known-but-unsupported fields.
pub fn reject_if_present(body: &Value, known_unsupported: &[&str]) -> Result<(), EdgeError> {
    for name in known_unsupported {
        if body.get(name).is_some() {
            return Err(EdgeError::unsupported(
                format!(
                    "field {name:?} is not implemented in this server's compatible subset; \
                     see the README for the exact list of supported parameters"
                ),
                name,
            ));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Responses
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Debug, Serialize)]
pub struct ModelObject {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub owned_by: &'static str,
}

#[derive(Debug, Serialize)]
pub struct ModelList {
    pub object: &'static str,
    pub data: Vec<ModelObject>,
}

#[derive(Debug, Serialize)]
pub struct ErrorDetail {
    pub message: String,
    pub r#type: &'static str,
    pub param: Option<String>,
    pub code: Option<&'static str>,
}

#[derive(Debug, Serialize)]
pub struct ErrorBody {
    pub error: ErrorDetail,
}

impl ErrorBody {
    pub fn from_error(err: &EdgeError) -> ErrorBody {
        let (r#type, param) = match err {
            EdgeError::InvalidRequest { param, .. } | EdgeError::Unsupported { param, .. } => {
                ("invalid_request_error", param.clone())
            }
            EdgeError::NoChatTemplate(_) => ("invalid_request_error", Some("messages".into())),
            EdgeError::NotFound(_) => ("invalid_request_error", Some("model".into())),
            EdgeError::Busy(_) => ("rate_limit_error", None),
            EdgeError::Engine(_) | EdgeError::Internal(_) => ("api_error", None),
        };
        ErrorBody {
            error: ErrorDetail {
                message: err.to_string(),
                r#type,
                param,
                code: Some(err.code()),
            },
        }
    }
}

/// HTTP status for each error kind (used by the server layer).
impl EdgeError {
    pub fn http_status(&self) -> u16 {
        match self {
            EdgeError::InvalidRequest { .. }
            | EdgeError::Unsupported { .. }
            | EdgeError::NoChatTemplate(_) => 400,
            EdgeError::NotFound(_) => 404,
            EdgeError::Busy(_) => 429,
            EdgeError::Engine(_) | EdgeError::Internal(_) => 500,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stop_accepts_string_and_array() {
        let p =
            validate_common(None, None, None, Some(StopField::One("END".into())), None).unwrap();
        assert_eq!(p.stop_strings, vec!["END"]);
        let p = validate_common(
            None,
            None,
            None,
            Some(StopField::Many(vec!["a".into(), "b".into()])),
            None,
        )
        .unwrap();
        assert_eq!(p.stop_strings, vec!["a", "b"]);
    }

    #[test]
    fn rejects_out_of_range_parameters() {
        let e = validate_common(Some(2.5), None, None, None, None).unwrap_err();
        assert!(e.to_string().contains("temperature"));
        let e = validate_common(None, Some(0.0), None, None, None).unwrap_err();
        assert!(e.to_string().contains("top_p"));
        let e = validate_common(None, None, Some(0), None, None).unwrap_err();
        assert!(e.to_string().contains("max_tokens"));
        let e = validate_common(None, None, None, Some(StopField::One(String::new())), None)
            .unwrap_err();
        assert!(e.to_string().contains("stop"));
        let e = validate_common(None, None, None, None, Some(-1)).unwrap_err();
        assert!(e.to_string().contains("seed"));
    }

    #[test]
    fn empty_stop_array_is_invalid_but_four_are_supported() {
        let e = validate_common(None, None, None, Some(StopField::Many(vec![])), None).unwrap_err();
        assert!(e.to_string().contains("at least one"));
        validate_common(
            None,
            None,
            None,
            Some(StopField::Many(vec![
                "a".into(),
                "b".into(),
                "c".into(),
                "d".into(),
            ])),
            None,
        )
        .unwrap();
        let e = validate_common(
            None,
            None,
            None,
            Some(StopField::Many((0..5).map(|i| i.to_string()).collect())),
            None,
        )
        .unwrap_err();
        assert_eq!(e.code(), "unsupported_parameter");
    }

    #[test]
    fn chat_message_rejects_multimodal_content_and_bad_roles() {
        let m = ChatMessage {
            role: "user".into(),
            content: Value::from("hi"),
        };
        assert_eq!(m.into_content().unwrap().0, "user");
        let m = ChatMessage {
            role: "user".into(),
            content: serde_json::json!([{"type": "text", "text": "hi"}]),
        };
        let err = m.into_content().unwrap_err();
        assert_eq!(err.code(), "invalid_value");
        let m = ChatMessage {
            role: "tool".into(),
            content: Value::from("x"),
        };
        assert!(m.into_content().is_err());
    }

    #[test]
    fn prompt_array_with_one_element_is_accepted() {
        let p = PromptField::Many(vec!["hello".into()]).into_text().unwrap();
        assert_eq!(p, "hello");
        let e = PromptField::Many(vec!["a".into(), "b".into()])
            .into_text()
            .unwrap_err();
        assert_eq!(e.code(), "unsupported_parameter");
    }

    #[test]
    fn temperature_zero_is_valid_greedy() {
        let p = validate_common(Some(0.0), None, None, None, None).unwrap();
        assert_eq!(p.temperature, 0.0);
    }
}
