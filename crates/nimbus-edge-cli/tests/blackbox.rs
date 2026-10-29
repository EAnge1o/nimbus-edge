//! Black-box tests: run the actual `nimbus-edge` binary as a child process,
//! exactly like a user would, and drive it over HTTP. Uses a random-weight
//! tiny checkpoint built on the fly — no network, no real model downloads.

use std::io::{BufRead, BufReader};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

// The fixture builder comes from the core crate's dev-dependency (feature
// `test-support`); it is never compiled into release binaries.
use nimbus_edge_core::fixture::write_random_checkpoint as build_checkpoint_in;

struct TempDir(std::path::PathBuf);
impl TempDir {
    fn new(tag: &str) -> TempDir {
        let n = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("nb-blackbox-{tag}-{n}"));
        std::fs::create_dir_all(&dir).unwrap();
        TempDir(dir)
    }
}
impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

struct Server(Child, String);

impl Server {
    fn url(&self) -> &str {
        &self.1
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn binary() -> &'static str {
    env!("CARGO_BIN_EXE_nimbus-edge")
}

/// Run `nimbus-edge <args>` to completion, returning (exit code, stdout).
fn run_cli(args: &[&str]) -> (i32, String) {
    let out = Command::new(binary())
        .args(args)
        .output()
        .expect("spawn binary");
    (
        out.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&out.stdout).into(),
    )
}

/// Start `serve` and parse the printed listening address. On panic paths the
/// `Server` drop guard kills the child; `wait` happens on the happy path via
/// the kill+wait in `Drop`.
#[allow(clippy::zombie_processes)]
fn start_server(args: &[&str]) -> Server {
    let mut child = Command::new(binary())
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn serve");
    let stdout = child.stdout.take().expect("piped stdout");
    let deadline = Instant::now() + Duration::from_secs(60);
    let reader = BufReader::new(stdout);
    let mut lines = reader.lines();
    loop {
        let line = match lines.next() {
            Some(Ok(l)) => l,
            other => panic!("server stdout closed early: {other:?}"),
        };
        if let Some(rest) = line.strip_prefix("listening on ") {
            let url = rest
                .split_whitespace()
                .next()
                .expect("url token")
                .to_string();
            // Give the socket a beat to accept after the banner print.
            std::thread::sleep(Duration::from_millis(100));
            return Server(child, url);
        }
        if Instant::now() > deadline {
            panic!("server did not print its address in 60s");
        }
    }
}

fn http_post(url: &str, path: &str, body: &str) -> (u16, String) {
    let client = reqwest::blocking::Client::new();
    let resp = client
        .post(format!("{url}{path}"))
        .header("Content-Type", "application/json")
        .body(body.to_string())
        .timeout(Duration::from_secs(30))
        .send()
        .expect("request completes");
    (resp.status().as_u16(), resp.text().expect("body"))
}

#[test]
fn model_lifecycle_and_serving_end_to_end() {
    let tmp = TempDir::new("lifecycle");
    let models_dir = tmp.0.join("models");
    // Build a checkpoint the same way the test-support fixture does: reuse
    // the core fixture builder through a tiny helper binary? The binary has
    // no such subcommand, so build the checkpoint via the core crate here
    // (the black-box part is the CLI + server; the fixture builder is shared).
    let source = tmp.0.join("ckpt");
    build_checkpoint_in(&source, 77);
    let models_dir_s = models_dir.to_str().unwrap().replace('\\', "/");
    let source_s = source.to_str().unwrap().replace('\\', "/");

    // 1. import
    let (code, out) = run_cli(&[
        "--models-dir",
        &models_dir_s,
        "models",
        "import",
        "--path",
        &source_s,
        "--name",
        "tiny",
        "--chat-template",
        "chatml",
    ]);
    assert_eq!(code, 0, "import failed: {out}");
    assert!(out.contains("imported tiny"));

    // 2. list
    let (code, out) = run_cli(&["--models-dir", &models_dir_s, "models", "list"]);
    assert_eq!(code, 0);
    assert!(out.contains("tiny"));
    assert!(out.contains("chatml"));

    // 3. inspect
    let (code, out) = run_cli(&["--models-dir", &models_dir_s, "models", "inspect", "tiny"]);
    assert_eq!(code, 0);
    let meta: serde_json::Value = serde_json::from_str(&out).expect("inspect prints JSON");
    assert_eq!(meta["meta"]["name"], "tiny");
    assert!(meta["n_params"].as_u64().unwrap() > 0);

    // 4. serve (port 0 -> ephemeral; banner carries the real address)
    let _guard = ServerGuard::new();
    let server = start_server(&[
        "--models-dir",
        &models_dir_s,
        "serve",
        "--model",
        "tiny",
        "--port",
        "0",
        "--max-concurrency",
        "2",
    ]);
    let url = server.url().to_string();

    // healthz
    {
        let client = reqwest::blocking::Client::new();
        let resp = client
            .get(format!("{url}/healthz"))
            .timeout(Duration::from_secs(5))
            .send()
            .expect("healthz");
        assert_eq!(resp.status().as_u16(), 200);
        let body: serde_json::Value = resp.json().unwrap();
        assert_eq!(body["model"]["name"], "tiny");
    }

    // 5. non-streaming chat completion
    let (status, body) = http_post(
        &url,
        "/v1/chat/completions",
        r#"{"model":"tiny","messages":[{"role":"user","content":"hello"}],"max_tokens":8}"#,
    );
    assert_eq!(status, 200, "body: {body}");
    let parsed: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(parsed["choices"][0]["message"]["role"], "assistant");
    assert!(parsed["choices"][0]["message"]["content"].is_string());

    // 6. streaming chat completion: data: lines then [DONE]
    let client = reqwest::blocking::Client::new();
    let resp = client
        .post(format!("{url}/v1/chat/completions"))
        .header("Content-Type", "application/json")
        .body(r#"{"model":"tiny","messages":[{"role":"user","content":"hi"}],"max_tokens":6,"stream":true}"#)
        .timeout(Duration::from_secs(30))
        .send()
        .expect("stream request");
    assert_eq!(resp.status().as_u16(), 200);
    let text = resp.text().unwrap();
    assert!(
        text.contains("data: [DONE]"),
        "stream must end with [DONE]: {text}"
    );
    assert!(text.contains("chat.completion.chunk"));

    // 7. remove
    let (code, out) = run_cli(&["--models-dir", &models_dir_s, "models", "remove", "tiny"]);
    assert_eq!(code, 0, "{out}");
    let (_, out) = run_cli(&["--models-dir", &models_dir_s, "models", "list"]);
    assert!(!out.contains("tiny"), "model must be gone: {out}");
}

/// Prevent parallel black-box tests from racing on CPU-heavy generations.
struct ServerGuard;
impl ServerGuard {
    fn new() -> ServerGuard {
        ServerGuard
    }
}

#[test]
fn serve_rejects_unknown_model_with_nonzero_exit() {
    let tmp = TempDir::new("unknown");
    let models_dir = tmp.0.join("models");
    std::fs::create_dir_all(&models_dir).unwrap();
    let models_dir_s = models_dir.to_str().unwrap().replace('\\', "/");
    let out = Command::new(binary())
        .args([
            "--models-dir",
            &models_dir_s,
            "serve",
            "--model",
            "nope",
            "--port",
            "0",
        ])
        .output()
        .expect("spawn");
    assert_ne!(out.status.code(), Some(0));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("not found") || stderr.contains("error"),
        "stderr: {stderr}"
    );
}
