//! Test fixture: builds a tiny random-weight GPT-2 checkpoint on disk so the
//! whole test suite (unit, integration, black-box, CI) runs without any
//! network access or real model downloads. The tokenizer is the vendored
//! tiny-GPT2 vocab from nimbus-infer-rs (plain text, 6 KB); weights are
//! generated from a seeded PRNG so generation is deterministic per seed.
//!
//! Enabled by the `test-support` feature; never compiled into release
//! binaries.

use std::fs;
use std::path::{Path, PathBuf};

pub const TINY_VOCAB: &str = include_str!("fixtures/tiny-vocab.json");
pub const TINY_MERGES: &str = include_str!("fixtures/tiny-merges.txt");

/// Checkpoint geometry. Small on purpose: a full test generation is
/// milliseconds.
pub const VOCAB: usize = 448;
pub const N_LAYERS: usize = 2;
pub const N_HEADS: usize = 4;
pub const D_MODEL: usize = 128;
pub const D_MLP: usize = 512;
pub const MAX_SEQ: usize = 256;

/// Seeded xorshift* PRNG (deterministic across platforms).
struct Rng(u64);

impl Rng {
    fn next_f32(&mut self) -> f32 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        let v = x.wrapping_mul(0x2545F4914F6CDD1D);
        ((v >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.08
    }
}

/// Write a complete random-weight checkpoint into `dir` and return the dir.
pub fn write_random_checkpoint(dir: &Path, seed: u64) -> PathBuf {
    let config = format!(
        r#"{{"vocab_size": {VOCAB}, "n_layers": {N_LAYERS}, "n_heads": {N_HEADS},
            "d_model": {D_MODEL}, "d_mlp": {D_MLP}, "max_seq_len": {MAX_SEQ},
            "head_dim": null, "n_kv_heads": null, "norm": "layernorm",
            "activation": "gelu", "pos_encoding": "learned", "tie_embeddings": true,
            "rope_theta": 10000.0, "norm_eps": 1e-05, "scale_attn_by_head_dim": true,
            "name": "tiny-random"}}"#
    );
    fs::create_dir_all(dir).expect("create checkpoint dir");
    fs::write(dir.join("config.json"), config).expect("write config.json");

    // Tensors, stored (in, out) like the engine expects. Norm weights sit
    // near 1.0/0.0 so random outputs stay in a sane numeric range.
    let mut rng = Rng(seed.wrapping_mul(0x9E3779B97F4A7C15) | 1);
    let mut tensors: Vec<(String, Vec<f32>, Vec<usize>)> = Vec::new();
    let near_one = |rng: &mut Rng| 1.0 + rng.next_f32();
    let mat = |rng: &mut Rng, rows: usize, cols: usize| {
        (0..rows * cols)
            .map(|_| rng.next_f32())
            .collect::<Vec<f32>>()
    };

    tensors.push((
        "wte".into(),
        mat(&mut rng, VOCAB, D_MODEL),
        vec![VOCAB, D_MODEL],
    ));
    tensors.push((
        "wpe".into(),
        mat(&mut rng, MAX_SEQ, D_MODEL),
        vec![MAX_SEQ, D_MODEL],
    ));
    for l in 0..N_LAYERS {
        let p = move |s: &str| format!("h.{l}.{s}");
        for name in [
            "attn.q.weight",
            "attn.k.weight",
            "attn.v.weight",
            "attn.o.weight",
        ] {
            tensors.push((
                p(name),
                mat(&mut rng, D_MODEL, D_MODEL),
                vec![D_MODEL, D_MODEL],
            ));
        }
        tensors.push((
            p("mlp.fc.weight"),
            mat(&mut rng, D_MODEL, D_MLP),
            vec![D_MODEL, D_MLP],
        ));
        tensors.push((
            p("mlp.proj.weight"),
            mat(&mut rng, D_MLP, D_MODEL),
            vec![D_MLP, D_MODEL],
        ));
        tensors.push((
            p("norm1.weight"),
            (0..D_MODEL).map(|_| near_one(&mut rng)).collect(),
            vec![D_MODEL],
        ));
        tensors.push((p("norm1.bias"), vec![0.0; D_MODEL], vec![D_MODEL]));
        tensors.push((
            p("norm2.weight"),
            (0..D_MODEL).map(|_| near_one(&mut rng)).collect(),
            vec![D_MODEL],
        ));
        tensors.push((p("norm2.bias"), vec![0.0; D_MODEL], vec![D_MODEL]));
    }
    tensors.push((
        "norm_f.weight".into(),
        (0..D_MODEL).map(|_| near_one(&mut rng)).collect(),
        vec![D_MODEL],
    ));
    tensors.push(("norm_f.bias".into(), vec![0.0; D_MODEL], vec![D_MODEL]));

    write_safetensors(&dir.join("weights.safetensors"), &tensors);

    let tok_dir = dir.join("tokenizer");
    fs::create_dir_all(&tok_dir).expect("create tokenizer dir");
    fs::write(tok_dir.join("vocab.json"), TINY_VOCAB).expect("write vocab.json");
    fs::write(tok_dir.join("merges.txt"), TINY_MERGES).expect("write merges.txt");
    dir.to_path_buf()
}

/// Minimal safetensors writer: `u64 LE header length` + JSON header + F32 LE
/// payloads, offsets packed in order.
pub fn write_safetensors(path: &Path, tensors: &[(String, Vec<f32>, Vec<usize>)]) {
    let mut data = Vec::new();
    let mut header = serde_json::Map::new();
    for (name, values, shape) in tensors {
        let start = data.len();
        for v in values {
            data.extend_from_slice(&v.to_le_bytes());
        }
        header.insert(
            name.clone(),
            serde_json::json!({
                "dtype": "F32",
                "shape": shape,
                "data_offsets": [start, data.len()],
            }),
        );
    }
    let header_text = serde_json::to_string(&serde_json::Value::Object(header)).unwrap();
    let mut file = Vec::with_capacity(8 + header_text.len() + data.len());
    file.extend_from_slice(&(header_text.len() as u64).to_le_bytes());
    file.extend_from_slice(header_text.as_bytes());
    file.extend_from_slice(&data);
    fs::write(path, file).expect("write weights.safetensors");
}

#[cfg(test)]
mod tests {
    use super::*;
    use nimbus_infer_rs::engine::QuantMode;

    #[test]
    fn checkpoint_is_loadable_and_deterministic() {
        let dir = std::env::temp_dir().join("nb-fixture-test");
        let _ = fs::remove_dir_all(&dir);
        write_random_checkpoint(&dir, 7);
        let engine = nimbus_infer_rs::engine::Engine::load(&dir, QuantMode::F32)
            .expect("fixture loads with the real engine");
        let prompt = engine.encode("hello world").unwrap();
        let params = nimbus_infer_rs::sampler::SamplingParams {
            max_new_tokens: 8,
            temperature: 0.0,
            ..Default::default()
        };
        let a = engine.generate_ids(&prompt, &params, None).unwrap();
        let b = engine.generate_ids(&prompt, &params, None).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.len(), 8);
        let _ = fs::remove_dir_all(&dir);
    }
}
