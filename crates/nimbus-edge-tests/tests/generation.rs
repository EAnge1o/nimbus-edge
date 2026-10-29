//! Prefix-cache correctness: hits must be *exactly* equivalent to fresh
//! prefill, misses must never produce a wrong reuse, and counters must move.

use nimbus_edge_core::backend::{Backend, GenRequest, NimbusBackend, PrefixCacheConfig};
use nimbus_edge_core::chat::ChatTemplate;
use nimbus_edge_core::fixture::write_random_checkpoint;
use nimbus_edge_core::openai::GenParams;
use nimbus_edge_tests::common::TempDir;

fn params(max_new_tokens: u32, seed: Option<u64>) -> GenRequest {
    GenRequest {
        prompt: String::new(),
        params: GenParams {
            max_new_tokens,
            temperature: 0.0,
            top_p: 1.0,
            seed,
            stop_strings: Vec::new(),
        },
    }
}

fn backend(dir: &std::path::Path, enabled: bool) -> NimbusBackend {
    NimbusBackend::load(
        dir,
        nimbus_infer_rs::engine::QuantMode::F32,
        "tiny".into(),
        ChatTemplate::None,
        PrefixCacheConfig {
            enabled,
            min_prompt_tokens: 4,
            ..Default::default()
        },
    )
    .unwrap()
}

/// The warm prompt must not end on whitespace: GPT-2 pre-tokenization
/// attaches a trailing space to the *next* word, which would break
/// token-prefix equality for the extension. Words separated by single spaces
/// keep `encode(a + b)` starting with `encode(a)` exactly.
const WARM_PROMPT: &str = "alpha beta gamma delta epsilon zeta eta theta iota";

#[test]
fn tokenizer_prefix_property_holds_for_the_fixture_prompts() {
    // The cache hit in the tests below relies on this property: extending a
    // prompt that does not end on whitespace keeps its token sequence as a
    // strict prefix.
    let tmp = TempDir::new("probe");
    let dir = write_random_checkpoint(&tmp.path().join("m"), 31);
    let engine =
        nimbus_infer_rs::engine::Engine::load(&dir, nimbus_infer_rs::engine::QuantMode::F32)
            .unwrap();
    let tok = engine.tokenizer.as_ref().unwrap();
    let ids1 = tok.encode(WARM_PROMPT);
    let extended = format!("{WARM_PROMPT} kappa lambda mu nu xi omicron pi");
    let ids2 = tok.encode(&extended);
    assert!(ids1.len() < ids2.len());
    assert_eq!(
        ids2[..ids1.len()],
        ids1[..],
        "token prefix must be preserved"
    );
}

#[test]
fn cached_prefix_produces_identical_output() {
    let tmp = TempDir::new("pcache");
    let dir = write_random_checkpoint(&tmp.path().join("m"), 31);

    // Baseline: no cache at all.
    let plain = backend(&dir, false);
    let mut baseline_req = params(16, None);
    baseline_req.prompt = WARM_PROMPT.to_string();
    let _ = plain
        .generate(&baseline_req, &|| false, &mut |_| {})
        .unwrap();

    // Cached backend: warm run misses and inserts one prompt-KV entry; the
    // extended prompt (multi-turn pattern) then hits it.
    let cached = backend(&dir, true);
    let mut warm = params(4, None);
    warm.prompt = WARM_PROMPT.to_string();
    cached.generate(&warm, &|| false, &mut |_| {}).unwrap();
    let stats_after_warm = cached.cache_stats();
    assert_eq!(stats_after_warm.inserts, 1, "warm run inserts one entry");

    let extended = format!("{WARM_PROMPT} kappa lambda mu nu xi omicron pi");
    let mut req2 = params(16, None);
    req2.prompt = extended;
    let via_cache = cached.generate(&req2, &|| false, &mut |_| {}).unwrap();
    let via_plain = plain.generate(&req2, &|| false, &mut |_| {}).unwrap();

    assert_eq!(
        via_cache.text, via_plain.text,
        "cache hit must not change output"
    );
    assert_eq!(via_cache.finish_reason, via_plain.finish_reason);
    assert_eq!(via_cache.prompt_tokens, via_plain.prompt_tokens);
    let stats = cached.cache_stats();
    assert_eq!(
        stats.hits, 1,
        "extended prompt hits the cached prefix (stats: {stats:?})"
    );
    assert_eq!(stats.misses, 1, "warm run was a miss");
}

#[test]
fn unrelated_prompts_miss_and_stay_correct() {
    let tmp = TempDir::new("pcache-miss");
    let dir = write_random_checkpoint(&tmp.path().join("m"), 32);
    let cached = backend(&dir, true);
    let plain = backend(&dir, false);

    let prompts = [
        "first completely different prompt with enough tokens to cache it",
        "second prompt sharing nothing at all with the first one here",
    ];
    for p in prompts {
        let mut r = params(12, None);
        r.prompt = p.to_string();
        let a = cached.generate(&r, &|| false, &mut |_| {}).unwrap();
        let b = plain.generate(&r, &|| false, &mut |_| {}).unwrap();
        assert_eq!(
            a.text, b.text,
            "outputs with and without cache must match for {p:?}"
        );
    }
    let stats = cached.cache_stats();
    assert_eq!(stats.misses, 2, "unrelated prompts never hit");
    assert_eq!(stats.inserts, 2);
    assert_eq!(stats.entries, 2);
}

#[test]
fn repeated_identical_prompts_never_false_reuse() {
    let tmp = TempDir::new("pcache-same");
    let dir = write_random_checkpoint(&tmp.path().join("m"), 33);
    let cached = backend(&dir, true);
    let mut r = params(10, None);
    r.prompt = "repeat me exactly the same way every time ok".to_string();

    let first = cached.generate(&r.clone(), &|| false, &mut |_| {}).unwrap();
    // Second identical request: the cached KV covers exactly this prompt, so
    // it cannot serve as a next-token source for itself — no hit, and the
    // output is still identical.
    let second = cached.generate(&r, &|| false, &mut |_| {}).unwrap();
    assert_eq!(first.text, second.text);
    assert_eq!(second.prompt_tokens, first.prompt_tokens);
    assert_eq!(second.finish_reason, first.finish_reason);
}

#[test]
fn cache_disabled_by_config() {
    let tmp = TempDir::new("pcache-off");
    let dir = write_random_checkpoint(&tmp.path().join("m"), 34);
    let off = backend(&dir, false);
    let mut r = params(8, None);
    r.prompt = "a prompt long enough to be cacheable if enabled".to_string();
    off.generate(&r, &|| false, &mut |_| {}).unwrap();
    let stats = off.cache_stats();
    assert_eq!(stats.inserts, 0);
    assert_eq!(stats.entries, 0);
}

#[test]
fn stop_string_truncates_text_and_counts_finish_stop() {
    let tmp = TempDir::new("stop");
    let dir = write_random_checkpoint(&tmp.path().join("m"), 35);
    let be = backend(&dir, false);
    let mut r = params(64, None);
    r.prompt = "generate a lot of random tiny-model text now please".to_string();
    let raw = be.generate(&r, &|| false, &mut |_| {}).unwrap();
    // Cut on the first three characters of the raw output: the stop string
    // then appears mid-stream (spanning deltas is the interesting case).
    let cut: String = raw.text.chars().take(3).collect();
    assert!(!cut.is_empty());
    r.params.stop_strings = vec![cut.clone()];
    let stopped = be.generate(&r, &|| false, &mut |_| {}).unwrap();
    assert_eq!(
        stopped.finish_reason,
        nimbus_edge_core::backend::FinishReason::Stop
    );
    assert!(
        !stopped.text.contains(&cut),
        "output must be cut before the stop sequence: {:?} vs cut {:?}",
        stopped.text,
        cut
    );
    assert!(stopped.text.len() < raw.text.len());
    assert!(stopped.completion_tokens <= raw.completion_tokens);
}
