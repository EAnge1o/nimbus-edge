//! HTTP routes: /healthz, /v1/models, /v1/chat/completions, /v1/completions.

use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::rejection::JsonRejection;
use axum::extract::State;
use axum::middleware;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};

use nimbus_edge_core::backend::GenRequest;
use nimbus_edge_core::chat::{ChatTemplate, RenderedMessage};
use nimbus_edge_core::error::EdgeError;
use nimbus_edge_core::openai::{
    validate_common, ChatCompletionRequest, ChatMessage, CompletionRequest, ModelList, ModelObject,
};
use nimbus_edge_core::scheduler::StreamEvent;

use crate::auth::require_bearer;
use crate::errors::{rejection, ApiError};
use crate::sse::{self, CancelAwareStream};
use crate::state::SharedState;

pub fn router(state: SharedState) -> Router {
    let v1 = Router::new()
        .route("/models", get(models))
        .route("/chat/completions", post(chat_completions))
        .route("/completions", post(completions))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            require_bearer,
        ));
    Router::new()
        .route("/healthz", get(healthz))
        .nest("/v1", v1)
        .with_state(state)
}

// ---------------------------------------------------------------------------
// health + models
// ---------------------------------------------------------------------------

async fn healthz(State(state): State<SharedState>) -> Json<serde_json::Value> {
    let info = &state.model;
    Json(serde_json::json!({
        "status": "ok",
        "version": nimbus_edge_core::VERSION,
        "uptime_seconds": state.started.elapsed().as_secs(),
        "model": {
            "name": info.name,
            "arch": info.arch,
            "quant": info.quant,
            "chat_template": info.chat_template,
            "context_len": info.context_len,
            "vocab_size": info.vocab_size,
            "tokenizer_vocab": info.tokenizer_vocab,
        },
        "queue": state.scheduler.stats(),
        "prefix_cache": state.scheduler.cache_stats(),
    }))
}

async fn models(State(state): State<SharedState>) -> Json<ModelList> {
    Json(ModelList {
        object: "list",
        data: vec![ModelObject {
            id: state.model.name.clone(),
            object: "model",
            created: unix_now(),
            owned_by: "nimbus-edge",
        }],
    })
}

// ---------------------------------------------------------------------------
// chat completions
// ---------------------------------------------------------------------------

/// Known fields of the wider OpenAI API that this subset deliberately does
/// not implement. They get an explicit 400, never silent ignoring.
const CHAT_UNSUPPORTED: &[&str] = &[
    "n",
    "logprobs",
    "top_logprobs",
    "presence_penalty",
    "frequency_penalty",
    "logit_bias",
    "tools",
    "tool_choice",
    "response_format",
    "stream_options",
    "max_completion_tokens",
    "service_tier",
    "metadata",
    "store",
];

const COMPLETION_UNSUPPORTED: &[&str] = &[
    "n",
    "logprobs",
    "top_logprobs",
    "presence_penalty",
    "frequency_penalty",
    "logit_bias",
    "echo",
    "suffix",
    "best_of",
    "stream_options",
    "service_tier",
    "metadata",
    "store",
];

fn invalid_body(detail: impl std::fmt::Display) -> ApiError {
    ApiError(EdgeError::InvalidRequest {
        message: format!(
            "invalid request body: {detail}; this server implements a documented \
             OpenAI-compatible subset and rejects unknown or invalid fields"
        ),
        param: None,
        code: "invalid_request_body",
    })
}

fn model_mismatch(requested: &str, served: &str) -> ApiError {
    ApiError(EdgeError::not_found(format!(
        "the model {requested:?} does not exist; this server serves {served:?} (see /v1/models)"
    )))
}

async fn chat_completions(
    State(state): State<SharedState>,
    body: Result<Json<serde_json::Value>, JsonRejection>,
) -> Response {
    let Json(raw) = match body {
        Ok(b) => b,
        Err(rej) => return rejection(rej).into_response(),
    };
    // Known-but-unsupported fields first: they must produce the dedicated
    // error, not serde's generic "unknown field".
    if let Err(e) = nimbus_edge_core::openai::reject_if_present(&raw, CHAT_UNSUPPORTED) {
        return ApiError(e).into_response();
    }
    let req: ChatCompletionRequest = match serde_json::from_value(raw) {
        Ok(r) => r,
        Err(e) => return invalid_body(e).into_response(),
    };
    if req.model != state.model.name {
        return model_mismatch(&req.model, &state.model.name).into_response();
    }

    let template = match ChatTemplate::parse(&state.model.chat_template) {
        Ok(t) => t,
        Err(e) => return ApiError(e).into_response(),
    };
    let mut messages = Vec::with_capacity(req.messages.len());
    for m in req.messages {
        match ChatMessage::into_content(m) {
            Ok((role, content)) => messages.push(RenderedMessage { role, content }),
            Err(e) => return ApiError(e).into_response(),
        }
    }
    if messages.is_empty() {
        return ApiError(EdgeError::invalid_param(
            "messages must contain at least one message",
            "messages",
        ))
        .into_response();
    }

    let params = match validate_common(
        req.temperature,
        req.top_p,
        req.max_tokens,
        req.stop,
        req.seed,
    ) {
        Ok(mut p) => {
            p.stop_strings.extend(template.stop_sequences());
            p
        }
        Err(e) => return ApiError(e).into_response(),
    };

    let prompt = match template.render(&messages) {
        Ok(p) => p,
        Err(e) => return ApiError(e).into_response(),
    };

    respond(state, GenRequest { prompt, params }, req.stream, true).await
}

// ---------------------------------------------------------------------------
// completions
// ---------------------------------------------------------------------------

async fn completions(
    State(state): State<SharedState>,
    body: Result<Json<serde_json::Value>, JsonRejection>,
) -> Response {
    let Json(raw) = match body {
        Ok(b) => b,
        Err(rej) => return rejection(rej).into_response(),
    };
    if let Err(e) = nimbus_edge_core::openai::reject_if_present(&raw, COMPLETION_UNSUPPORTED) {
        return ApiError(e).into_response();
    }
    let req: CompletionRequest = match serde_json::from_value(raw) {
        Ok(r) => r,
        Err(e) => return invalid_body(e).into_response(),
    };
    if req.model != state.model.name {
        return model_mismatch(&req.model, &state.model.name).into_response();
    }
    let prompt = match req.prompt.into_text() {
        Ok(p) => p,
        Err(e) => return ApiError(e).into_response(),
    };
    let params = match validate_common(
        req.temperature,
        req.top_p,
        req.max_tokens,
        req.stop,
        req.seed,
    ) {
        Ok(p) => p,
        Err(e) => return ApiError(e).into_response(),
    };
    respond(state, GenRequest { prompt, params }, req.stream, false).await
}

// ---------------------------------------------------------------------------
// shared pipeline
// ---------------------------------------------------------------------------

/// Submit to the scheduler and shape the response (JSON or SSE).
async fn respond(state: SharedState, req: GenRequest, stream: bool, is_chat: bool) -> Response {
    let handle = match state.scheduler.submit(req) {
        Ok(h) => h,
        Err(e) => return ApiError(e).into_response(),
    };
    let id = format!("chatcmpl-{:016x}", handle.id);
    let model = state.model.name.clone();

    if stream {
        stream_response(handle, id, model).await
    } else {
        collect_json(handle, id, model, is_chat).await
    }
}

/// Non-streaming: await Done and emit the OpenAI JSON object. The handler
/// owns both receivers, so a client disconnect (hyper cancels the handler
/// future) cancels the generation as well.
async fn collect_json(
    handle: nimbus_edge_core::scheduler::SubmitHandle,
    id: String,
    model: String,
    is_chat: bool,
) -> Response {
    let mut rx = handle.rx;
    let _alive = handle.alive;
    let mut text = String::new();
    loop {
        match rx.recv().await {
            Some(StreamEvent::Token(delta)) => text.push_str(&delta),
            Some(StreamEvent::Done(Ok(output))) => {
                let usage = serde_json::json!({
                    "prompt_tokens": output.prompt_tokens,
                    "completion_tokens": output.completion_tokens,
                    "total_tokens": output.prompt_tokens + output.completion_tokens,
                });
                let choice = if is_chat {
                    serde_json::json!({
                        "index": 0,
                        "message": { "role": "assistant", "content": output.text },
                        "finish_reason": output.finish_reason.as_str(),
                    })
                } else {
                    serde_json::json!({
                        "index": 0,
                        "text": output.text,
                        "finish_reason": output.finish_reason.as_str(),
                        "logprobs": null,
                    })
                };
                return Json(serde_json::json!({
                    "id": id,
                    "object": if is_chat { "chat.completion" } else { "text_completion" },
                    "created": unix_now(),
                    "model": model,
                    "choices": [choice],
                    "usage": usage,
                }))
                .into_response();
            }
            Some(StreamEvent::Done(Err(e))) => return ApiError(e).into_response(),
            None => {
                return ApiError(EdgeError::Internal(
                    "generation ended without a terminal event".into(),
                ))
                .into_response()
            }
        }
    }
}

/// Streaming: shape scheduler events into `chat.completion.chunk` SSE events.
///
/// The shaped stream owns the scheduler's event receiver *and* the liveness
/// receiver. When the client disconnects, axum drops the response body, both
/// receivers drop, and the scheduler stops the generation at the next decode
/// step.
async fn stream_response(
    handle: nimbus_edge_core::scheduler::SubmitHandle,
    id: String,
    model: String,
) -> Response {
    let events = CancelAwareStream::new(handle.rx, handle.alive);
    let created = unix_now();
    let shaped = sse::ShapedChatStream::new(events, id, model, created);
    axum::response::Sse::new(sse::with_done(shaped))
        .keep_alive(sse::keep_alive())
        .into_response()
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}
