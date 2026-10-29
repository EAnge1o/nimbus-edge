//! API integration tests: request parsing, error responses, non-streaming
//! and streaming completions against the real HTTP stack.

use nimbus_edge_tests::common;

use common::{chat_body, spawn_server, ServerOptions, TempDir, TestServer};
use nimbus_edge_core::chat::ChatTemplate;

async fn server(template: ChatTemplate) -> (TempDir, TestServer) {
    let tmp = TempDir::new("api");
    let model = tmp.path().join("model");
    let _ = std::fs::create_dir_all(&model);
    nimbus_edge_core::fixture::write_random_checkpoint(&model, 11);
    let srv = spawn_server(
        &model,
        ServerOptions {
            template,
            ..Default::default()
        },
    )
    .await;
    (tmp, srv)
}

#[tokio::test]
async fn completions_non_stream_returns_openai_shape() {
    let (_t, srv) = server(ChatTemplate::None).await;
    let (status, body) = srv
        .post(
            "/v1/completions",
            serde_json::json!({"model":"tiny","prompt":"the engine caches keys and values","max_tokens":8}),
        )
        .await;
    assert_eq!(status, 200, "body: {body}");
    assert_eq!(body["object"], "text_completion");
    assert_eq!(body["model"], "tiny");
    let choice = &body["choices"][0];
    assert!(choice["text"].is_string());
    assert_eq!(choice["finish_reason"], "length");
    let usage = &body["usage"];
    assert!(usage["prompt_tokens"].as_u64().unwrap() > 0);
    assert_eq!(usage["completion_tokens"].as_u64().unwrap(), 8);
    assert_eq!(
        usage["total_tokens"],
        usage["prompt_tokens"].as_u64().unwrap() + usage["completion_tokens"].as_u64().unwrap()
    );
}

#[tokio::test]
async fn completions_deterministic_across_calls() {
    let (_t, srv) = server(ChatTemplate::None).await;
    let body = serde_json::json!({"model":"tiny","prompt":"abc def","max_tokens":10,"seed":7,"temperature":0});
    let (_, a) = srv.post("/v1/completions", body.clone()).await;
    let (_, b) = srv.post("/v1/completions", body).await;
    assert_eq!(a["choices"][0]["text"], b["choices"][0]["text"]);
}

#[tokio::test]
async fn chat_with_template_returns_assistant_message() {
    let (_t, srv) = server(ChatTemplate::ChatMl).await;
    let (status, body) = srv
        .post("/v1/chat/completions", chat_body("hello", 8))
        .await;
    assert_eq!(status, 200, "body: {body}");
    assert_eq!(body["object"], "chat.completion");
    assert_eq!(body["choices"][0]["message"]["role"], "assistant");
    assert!(body["choices"][0]["message"]["content"].is_string());
    // The ChatML stop sequence keeps the reply clean.
    assert_eq!(body["choices"][0]["finish_reason"], "length");
}

#[tokio::test]
async fn chat_without_template_is_a_clear_error_not_invented_prompt() {
    let (_t, srv) = server(ChatTemplate::None).await;
    let (status, body) = srv
        .post("/v1/chat/completions", chat_body("hello", 8))
        .await;
    assert_eq!(status, 400);
    let err = &body["error"];
    assert_eq!(err["code"], "no_chat_template");
    assert!(err["message"]
        .as_str()
        .unwrap()
        .contains("no chat template"));
    assert_eq!(err["type"], "invalid_request_error");
}

#[tokio::test]
async fn streaming_emits_role_first_then_done() {
    let (_t, srv) = server(ChatTemplate::ChatMl).await;
    let mut body = chat_body("hello", 8);
    body["stream"] = serde_json::json!(true);
    let events = srv.collect_sse("/v1/chat/completions", body).await;
    assert_eq!(
        events.last().map(String::as_str),
        Some("[DONE]"),
        "events: {events:?}"
    );
    let chunks: Vec<serde_json::Value> = events[..events.len() - 1]
        .iter()
        .map(|e| serde_json::from_str(e).expect("each data line is JSON"))
        .collect();
    assert!(
        chunks.len() >= 2,
        "expected role+content+finish chunks, got {events:?}"
    );
    for c in &chunks {
        assert_eq!(c["object"], "chat.completion.chunk");
        assert_eq!(c["model"], "tiny");
    }
    let first = &chunks[0]["choices"][0]["delta"];
    assert_eq!(first["role"], "assistant");
    assert!(first["content"].is_string());
    let last = chunks.last().unwrap()["choices"][0].clone();
    assert_eq!(last["finish_reason"], "length");
    assert_eq!(last["delta"]["content"], serde_json::Value::Null);
    // Content deltas concatenate to a string.
    let mut text = String::new();
    for c in &chunks[..chunks.len() - 1] {
        if let Some(s) = c["choices"][0]["delta"]["content"].as_str() {
            text.push_str(s);
        }
    }
    assert!(!text.is_empty());
}

#[tokio::test]
async fn unknown_fields_are_rejected_not_ignored() {
    let (_t, srv) = server(ChatTemplate::None).await;
    let (status, body) = srv
        .post(
            "/v1/completions",
            serde_json::json!({"model":"tiny","prompt":"x","max_tokens":4,"logit_bias":{}}),
        )
        .await;
    assert_eq!(status, 400);
    assert_eq!(body["error"]["code"], "unsupported_parameter");
    assert_eq!(body["error"]["param"], "logit_bias");
}

#[tokio::test]
async fn truly_unknown_fields_reject_with_field_name() {
    let (_t, srv) = server(ChatTemplate::None).await;
    let (status, body) = srv
        .post(
            "/v1/completions",
            serde_json::json!({"model":"tiny","prompt":"x","max_tokens":4,"banana":1}),
        )
        .await;
    assert_eq!(status, 400);
    assert_eq!(body["error"]["code"], "invalid_request_body");
    assert!(body["error"]["message"]
        .as_str()
        .unwrap()
        .contains("banana"));
}

#[tokio::test]
async fn out_of_range_parameters_get_clear_errors() {
    let (_t, srv) = server(ChatTemplate::None).await;
    for (field, value) in [
        ("temperature", serde_json::json!(3.0)),
        ("top_p", serde_json::json!(0.0)),
        ("max_tokens", serde_json::json!(0)),
        ("seed", serde_json::json!(-5)),
    ] {
        let mut body = serde_json::json!({"model":"tiny","prompt":"x","max_tokens":4});
        body[field] = value;
        let (status, resp) = srv.post("/v1/completions", body).await;
        assert_eq!(status, 400, "field {field}");
        assert_eq!(resp["error"]["param"], field, "field {field}: {resp}");
    }
    // Five stop sequences exceed the documented OpenAI limit of 4.
    let (status, resp) = srv
        .post(
            "/v1/completions",
            serde_json::json!({"model":"tiny","prompt":"x","max_tokens":4,
                "stop":["a","b","c","d","e"]}),
        )
        .await;
    assert_eq!(status, 400);
    assert_eq!(resp["error"]["param"], "stop");
}

#[tokio::test]
async fn wrong_model_name_is_404_model_not_found() {
    let (_t, srv) = server(ChatTemplate::None).await;
    let (status, body) = srv
        .post(
            "/v1/completions",
            serde_json::json!({"model":"gpt-4","prompt":"x","max_tokens":4}),
        )
        .await;
    assert_eq!(status, 404);
    assert!(body["error"]["message"].as_str().unwrap().contains("gpt-4"));
}

#[tokio::test]
async fn models_endpoint_lists_the_served_model() {
    let (_t, srv) = server(ChatTemplate::None).await;
    let (status, body) = srv.get("/v1/models").await;
    assert_eq!(status, 200);
    assert_eq!(body["object"], "list");
    assert_eq!(body["data"][0]["id"], "tiny");
    assert_eq!(body["data"][0]["object"], "model");
}

#[tokio::test]
async fn healthz_reports_queue_and_model() {
    let (_t, srv) = server(ChatTemplate::None).await;
    let (status, body) = srv.get("/healthz").await;
    assert_eq!(status, 200);
    assert_eq!(body["status"], "ok");
    assert_eq!(body["model"]["name"], "tiny");
    assert_eq!(body["model"]["context_len"], 256);
    assert!(body["queue"]["max_concurrency"].is_u64());
}

#[tokio::test]
async fn prompt_longer_than_context_is_rejected() {
    let (_t, srv) = server(ChatTemplate::None).await;
    let long_prompt = "word ".repeat(400); // way beyond 256-token context
    let (status, body) = srv
        .post(
            "/v1/completions",
            serde_json::json!({"model":"tiny","prompt":long_prompt,"max_tokens":4}),
        )
        .await;
    assert_eq!(status, 400);
    assert!(body["error"]["message"]
        .as_str()
        .unwrap()
        .contains("context"));
}

#[tokio::test]
async fn malformed_json_gets_a_parseable_error() {
    let (_t, srv) = server(ChatTemplate::None).await;
    let resp = srv
        .client
        .post(format!("{}/v1/completions", srv.base))
        .header("Content-Type", "application/json")
        .body("{not json")
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status().as_u16(), 400);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["error"]["type"], "invalid_request_error");
}

#[tokio::test]
async fn bearer_token_is_enforced_on_v1_but_not_healthz() {
    let tmp = TempDir::new("auth");
    let model = tmp.path().join("model");
    nimbus_edge_core::fixture::write_random_checkpoint(&model, 3);
    let srv = spawn_server(
        &model,
        ServerOptions {
            auth_token: Some("secret-token".into()),
            ..Default::default()
        },
    )
    .await;
    let (status, _) = srv.get("/healthz").await;
    assert_eq!(status, 200, "healthz stays open");

    let resp = srv
        .client
        .get(format!("{}/v1/models", srv.base))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status().as_u16(), 401);

    let resp = srv
        .client
        .get(format!("{}/v1/models", srv.base))
        .header("Authorization", "Bearer wrong")
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status().as_u16(), 401);

    let resp = srv
        .client
        .get(format!("{}/v1/models", srv.base))
        .header("Authorization", "Bearer secret-token")
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status().as_u16(), 200);
}
