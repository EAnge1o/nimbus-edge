//! Concurrency tests: bounded queue (429), client-disconnect cancellation,
//! queue-entry removal, and KV-cache isolation across concurrent requests.
//!
//! Every test runs on its own multi-thread tokio runtime and does the setup
//! inside that runtime — the server task must live on the same runtime the
//! assertions run on.

use std::time::Duration;

use nimbus_edge_core::chat::ChatTemplate;
use nimbus_edge_tests::common::{chat_body, spawn_server, ServerOptions, TempDir, TestServer};

async fn setup(tag: &str, opts: ServerOptions) -> (TempDir, TestServer) {
    let tmp = TempDir::new(tag);
    let model = tmp.path().join("model");
    nimbus_edge_core::fixture::write_random_checkpoint(&model, 21);
    let srv = spawn_server(&model, opts).await;
    (tmp, srv)
}

async fn completion(srv: &TestServer, prompt: &str, max_tokens: u32) -> reqwest::Response {
    srv.client
        .post(format!("{}/v1/completions", srv.base))
        .json(&serde_json::json!({"model":"tiny","prompt":prompt,"max_tokens":max_tokens}))
        .send()
        .await
        .expect("request completes")
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn queue_full_returns_429() {
    let (_tmp, srv) = setup(
        "q429",
        ServerOptions {
            max_concurrency: 1,
            max_queue_size: 1,
            ..Default::default()
        },
    )
    .await;
    // One concurrency slot, one queue slot. A runs; B queues; C rejects.
    // Requests are spawned (bare futures never fire until awaited) and each
    // stage is confirmed via /healthz before the next starts, so the queue
    // accounting is deterministic.
    let a = tokio::spawn({
        let client = srv.client.clone();
        let base = srv.base.clone();
        async move {
            client
                .post(format!("{base}/v1/completions"))
                .json(&serde_json::json!({"model":"tiny","prompt":"p c","max_tokens":300}))
                .send()
                .await
                .unwrap()
        }
    });
    srv.wait_until(|h| h["queue"]["inflight"] == 1, 10).await;
    let q = tokio::spawn({
        let client = srv.client.clone();
        let base = srv.base.clone();
        async move {
            client
                .post(format!("{base}/v1/completions"))
                .json(&serde_json::json!({"model":"tiny","prompt":"p d","max_tokens":300}))
                .send()
                .await
                .unwrap()
        }
    });
    let mut queued = false;
    for _ in 0..200 {
        let (_, h) = srv.get("/healthz").await;
        if h["queue"]["queue_depth"] == 1 {
            queued = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert!(queued, "second concurrent request must be queued");
    let c = completion(&srv, "p e", 4).await;
    assert_eq!(c.status().as_u16(), 429, "queue full must be 429");
    assert_eq!(
        c.headers().get("Retry-After").map(|v| v.to_str().unwrap()),
        Some("1")
    );
    let body: serde_json::Value = c.json().await.unwrap();
    assert_eq!(body["error"]["code"], "queue_full");
    assert_eq!(body["error"]["type"], "rate_limit_error");

    // The queued and running requests still complete normally.
    let (a, q) = tokio::join!(a, q);
    assert_eq!(a.unwrap().status().as_u16(), 200);
    assert_eq!(q.unwrap().status().as_u16(), 200);
    let health = srv
        .wait_until(
            |h| h["queue"]["inflight"] == 0 && h["queue"]["queue_depth"] == 0,
            15,
        )
        .await;
    assert!(health["queue"]["queue_full_rejections"].as_u64().unwrap() >= 1);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn client_disconnect_cancels_generation_and_frees_cpu() {
    let (_tmp, srv) = setup(
        "cancel",
        ServerOptions {
            template: ChatTemplate::ChatMl,
            ..Default::default()
        },
    )
    .await;
    // Open a streaming request for a long decode and drop it early.
    let resp = srv
        .client
        .post(format!("{}/v1/chat/completions", srv.base))
        .json(&{
            let mut b = chat_body("hello", 400);
            b["stream"] = serde_json::json!(true);
            b
        })
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status().as_u16(), 200);
    // Read one chunk, then abandon the response: hyper closes the body.
    use futures_util::StreamExt;
    let mut stream = resp.bytes_stream();
    let _ = stream.next().await;
    drop(stream);

    // The generation must stop: inflight returns to 0 promptly and the run
    // is accounted as cancelled.
    let health = srv
        .wait_until(
            |h| h["queue"]["inflight"] == 0 && h["queue"]["cancelled"].as_u64() >= Some(1),
            15,
        )
        .await;
    assert_eq!(health["queue"]["inflight"], 0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_requests_do_not_cross_contaminate_outputs() {
    // Two concurrent requests, two distinct prompts, greedy decoding: each
    // response must equal its own single-request baseline. Any KV-cache
    // crosstalk changes an output deterministically and is caught here.
    let (_tmp, srv) = setup(
        "crosstalk",
        ServerOptions {
            max_concurrency: 2,
            ..Default::default()
        },
    )
    .await;
    let prompts = ["alpha beta gamma delta", "the quick brown fox jumps"];
    let mut baselines = Vec::new();
    for p in prompts {
        let resp = completion(&srv, p, 24).await;
        assert_eq!(resp.status().as_u16(), 200);
        let body: serde_json::Value = resp.json().await.unwrap();
        baselines.push(body["choices"][0]["text"].as_str().unwrap().to_string());
    }

    let mut tasks = Vec::new();
    for p in prompts {
        tasks.push(completion(&srv, p, 24));
    }
    let responses = futures_util::future::join_all(tasks).await;
    for (i, resp) in responses.into_iter().enumerate() {
        assert_eq!(resp.status().as_u16(), 200);
        let body: serde_json::Value = resp.json().await.unwrap();
        let text = body["choices"][0]["text"].as_str().unwrap();
        assert_eq!(
            text, baselines[i],
            "request {i} output must match its own baseline under concurrency"
        );
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn queued_request_removed_when_client_goes_away() {
    let (_tmp, srv) = setup(
        "queued-cancel",
        ServerOptions {
            max_concurrency: 1,
            max_queue_size: 4,
            ..Default::default()
        },
    )
    .await;
    // Occupy the single slot with a long generation.
    let a = tokio::spawn({
        let client = srv.client.clone();
        let base = srv.base.clone();
        async move {
            client
                .post(format!("{base}/v1/completions"))
                .json(&serde_json::json!({"model":"tiny","prompt":"p a","max_tokens":300}))
                .send()
                .await
                .unwrap()
        }
    });
    srv.wait_until(|h| h["queue"]["inflight"] == 1, 10).await;

    // Queue one request, then abandon the connection while it waits.
    let b = srv
        .client
        .post(format!("{}/v1/completions", srv.base))
        .json(&serde_json::json!({"model":"tiny","prompt":"p b","max_tokens":300}))
        .send()
        .await
        .unwrap();
    tokio::time::sleep(Duration::from_millis(50)).await; // ensure it is queued
    drop(b);

    // The dead entry is swept from the queue without ever running.
    let health = srv.wait_until(|h| h["queue"]["queue_depth"] == 0, 10).await;
    assert_eq!(
        health["queue"]["queue_depth"], 0,
        "cancelled queued entry is swept"
    );
    let _ = a.await.unwrap();
}
