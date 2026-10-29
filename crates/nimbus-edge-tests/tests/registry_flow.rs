//! Registry flow tests: import, list, inspect, remove, checksum, and the
//! path-traversal defenses.

use std::path::Path;

use nimbus_edge_core::chat::ChatTemplate;
use nimbus_edge_core::error::EdgeError;
use nimbus_edge_core::fixture::write_random_checkpoint;
use nimbus_edge_core::registry::{verify_checksums, Registry};
use nimbus_edge_tests::common::TempDir;

#[test]
fn import_list_inspect_remove_roundtrip() {
    let tmp = TempDir::new("reg");
    let registry = Registry::open(tmp.path().join("models"));
    let source = write_random_checkpoint(&tmp.path().join("ckpt"), 5);

    registry
        .import(
            &source,
            "tiny-a",
            ChatTemplate::ChatMl,
            "fp32",
            false,
            "2026-01-01T00:00:00Z".into(),
        )
        .expect("import");

    let listing = registry.list().unwrap();
    assert_eq!(listing.models.len(), 1);
    assert_eq!(listing.models[0].name, "tiny-a");
    assert_eq!(listing.models[0].chat_template, "chatml");
    assert_eq!(listing.models[0].arch, "gpt2");
    assert!(listing.invalid.is_empty());

    let report = registry.inspect("tiny-a").unwrap();
    assert!(report.n_params > 0);
    assert!(report.has_tokenizer);
    assert!(report.tensor_names.iter().any(|t| t == "wte"));
    assert!(report.tensor_names.iter().any(|t| t == "h.0.attn.q.weight"));

    // Duplicate import without --overwrite is refused.
    assert!(registry
        .import(
            &source,
            "tiny-a",
            ChatTemplate::None,
            "fp32",
            false,
            "2026-01-01T00:00:00Z".into()
        )
        .is_err());

    // --overwrite replaces the model (template changes here).
    registry
        .import(
            &source,
            "tiny-a",
            ChatTemplate::None,
            "fp32",
            true,
            "2026-01-01T00:00:00Z".into(),
        )
        .expect("overwrite");
    assert_eq!(registry.load_meta("tiny-a").unwrap().chat_template, "none");

    registry.remove("tiny-a").expect("remove");
    assert!(registry.list().unwrap().models.is_empty());
}

#[test]
fn import_rejects_non_checkpoints() {
    let tmp = TempDir::new("reg-bad");
    let registry = Registry::open(tmp.path().join("models"));

    let empty = tmp.path().join("empty-dir");
    std::fs::create_dir_all(&empty).unwrap();
    let err = registry
        .import(
            &empty,
            "m",
            ChatTemplate::None,
            "fp32",
            false,
            "2026-01-01T00:00:00Z".into(),
        )
        .unwrap_err();
    assert!(err.to_string().contains("config.json"), "got: {err}");

    let bad_config = tmp.path().join("bad-config");
    std::fs::create_dir_all(&bad_config).unwrap();
    // All required integer fields present, but the head counts are
    // inconsistent — the engine's config parser must reject it.
    std::fs::write(
        bad_config.join("config.json"),
        r#"{"vocab_size":8,"n_layers":1,"n_heads":12,"n_kv_heads":5,"d_model":8,"d_mlp":8,"max_seq_len":8}"#,
    )
    .unwrap();
    let err = registry
        .import(
            &bad_config,
            "m",
            ChatTemplate::None,
            "fp32",
            false,
            "2026-01-01T00:00:00Z".into(),
        )
        .unwrap_err();
    assert!(err.to_string().contains("rejected"), "got: {err}");

    let no_weights = tmp.path().join("no-weights");
    std::fs::create_dir_all(&no_weights).unwrap();
    std::fs::write(
        no_weights.join("config.json"),
        r#"{"vocab_size":8,"n_layers":1,"n_heads":1,"d_model":8,"d_mlp":8,"max_seq_len":8}"#,
    )
    .unwrap();
    let err = registry
        .import(
            &no_weights,
            "m",
            ChatTemplate::None,
            "fp32",
            false,
            "2026-01-01T00:00:00Z".into(),
        )
        .unwrap_err();
    assert!(
        err.to_string().contains("weights.safetensors"),
        "got: {err}"
    );
}

#[test]
fn invalid_model_names_and_paths_are_rejected() {
    let tmp = TempDir::new("reg-traversal");
    let registry = Registry::open(tmp.path().join("models"));

    for bad in ["..", "../escape", "a/b", "a\\b", ".hidden", ""] {
        let err = registry
            .load_meta(bad)
            .err()
            .unwrap_or_else(|| panic!("name {bad:?} must be rejected"));
        let _ = err;
        assert!(
            registry.remove(bad).is_err(),
            "remove with name {bad:?} must fail"
        );
    }
    // A valid name whose directory does not exist -> NotFound (not a panic).
    assert!(matches!(
        registry.load_meta("missing"),
        Err(EdgeError::NotFound(_))
    ));
}

#[test]
fn checksum_verification_detects_tampering() {
    let tmp = TempDir::new("reg-checksum");
    let registry = Registry::open(tmp.path().join("models"));
    let source = write_random_checkpoint(&tmp.path().join("ckpt"), 9);
    registry
        .import(
            &source,
            "tiny",
            ChatTemplate::None,
            "fp32",
            false,
            "2026-01-01T00:00:00Z".into(),
        )
        .unwrap();
    let dir = registry.root.join("tiny");
    let meta = registry.load_meta("tiny").unwrap();
    verify_checksums(&dir, &meta).expect("fresh import verifies");

    // Tamper with the weights after import.
    let weights = dir.join("weights.safetensors");
    let bytes = std::fs::read(&weights).unwrap();
    let mut tampered = bytes.clone();
    let mid = bytes.len() / 2;
    tampered[mid] ^= 0xFF;
    std::fs::write(&weights, &tampered).unwrap();
    let err = verify_checksums(&dir, &meta).unwrap_err();
    assert!(err.to_string().contains("checksum mismatch"), "got: {err}");
    std::fs::write(&weights, &bytes).unwrap();
}

#[test]
fn sidecar_meta_without_registry_is_accepted_for_direct_paths() {
    let tmp = TempDir::new("reg-direct");
    let registry = Registry::open(tmp.path().join("models"));
    let source = write_random_checkpoint(&tmp.path().join("ckpt"), 2);
    // Not imported: resolve by path.
    let loc = registry.resolve(source.to_str().unwrap()).unwrap();
    assert!(loc.meta().is_none(), "no sidecar yet");
    // Import writes the sidecar; resolving the SOURCE path reads it.
    registry
        .import(
            &source,
            "tiny",
            ChatTemplate::ChatMl,
            "fp32",
            false,
            "2026-01-01T00:00:00Z".into(),
        )
        .unwrap();
    // The sidecar lives in the registry copy, not the source; source still
    // has none, so template defaults to none at serve time.
    let loc = registry.resolve(source.to_str().unwrap()).unwrap();
    assert!(loc.meta().is_none());
    assert!(matches!(
        loc,
        nimbus_edge_core::registry::ModelLocation::Direct { .. }
    ));
}

#[test]
fn loading_engine_from_corrupted_weights_fails_clearly() {
    let tmp = TempDir::new("reg-corrupt");
    let dir = write_random_checkpoint(&tmp.path().join("ckpt"), 4);
    let weights = dir.join("weights.safetensors");
    let bytes = std::fs::read(&weights).unwrap();
    std::fs::write(&weights, &bytes[..bytes.len() / 3]).unwrap();
    let err = nimbus_infer_rs::engine::Engine::load(&dir, nimbus_infer_rs::engine::QuantMode::F32)
        .err()
        .expect("corrupted weights must fail to load");
    assert!(!err.is_empty());
    // And an empty directory is equally rejected.
    let err2 = nimbus_infer_rs::engine::Engine::load(
        Path::new(&tmp.path().join("nope")),
        nimbus_infer_rs::engine::QuantMode::F32,
    )
    .err()
    .expect("missing dir must fail");
    assert!(
        err2.contains("config.json") || err2.contains("no weights"),
        "got: {err2}"
    );
}
