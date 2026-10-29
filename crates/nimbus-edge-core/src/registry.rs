//! Local model registry.
//!
//! Models live under `<models-dir>/<name>/` and carry an `edge-meta.json`
//! sidecar with per-file SHA-256 checksums, size, architecture and the chat
//! template choice. Import validates the checkpoint layout
//! (`config.json` + `weights.safetensors`|`model.safetensors`, optional
//! `tokenizer/`) before copying, so a half-broken directory never enters the
//! registry. Model names are strictly validated and every registry access is
//! containment-checked against the registry root — no path traversal.
//!
//! The first-class format is nimbus-infer-rs' checkpoint
//! (`config.json + weights.safetensors + tokenizer/{vocab.json,merges.txt}`);
//! the [`ModelFormat`] enum is the extension point for future format
//! adapters (GGUF and friends).

use std::collections::BTreeMap;
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::chat::ChatTemplate;
use crate::error::EdgeError;

pub const META_FILE: &str = "edge-meta.json";
pub const FORMAT_TAG: &str = "nimbus-checkpoint-v1";
const COPY_BUFFER_BYTES: usize = 1 << 20;
const MAX_HEADER_BYTES: u64 = 64 << 20; // safetensors header sanity cap

// ---------------------------------------------------------------------------
// Metadata
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct FileMeta {
    pub sha256: String,
    pub bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ModelMeta {
    pub format: String,
    pub name: String,
    /// RFC 3339 UTC timestamp of the import.
    pub created_at: String,
    /// Where the checkpoint was imported from (informational).
    pub source: String,
    /// Chat template name: "none" | "chatml" | ...
    pub chat_template: String,
    /// "gpt2" | "llama" — derived from config.json at import time.
    pub arch: String,
    /// Quantization recorded for the import, e.g. "fp32". Load-time quant is
    /// a serve-time choice; this is the hint shown by `models list`.
    pub quant_hint: String,
    pub files: BTreeMap<String, FileMeta>,
}

impl ModelMeta {
    pub fn chat_template(&self) -> Result<ChatTemplate, EdgeError> {
        ChatTemplate::parse(&self.chat_template)
    }

    pub fn total_bytes(&self) -> u64 {
        self.files.values().map(|f| f.bytes).sum()
    }

    pub fn validate(&self, name: &str) -> Result<(), EdgeError> {
        if self.format != FORMAT_TAG {
            return Err(EdgeError::invalid(format!(
                "unsupported model metadata format {:?} (expected {FORMAT_TAG:?})",
                self.format
            )));
        }
        if self.name != name {
            return Err(EdgeError::invalid(format!(
                "metadata name {:?} does not match directory name {name:?}",
                self.name
            )));
        }
        ChatTemplate::parse(&self.chat_template)?;
        if !matches!(self.arch.as_str(), "gpt2" | "llama") {
            return Err(EdgeError::invalid(format!(
                "unknown architecture {:?}",
                self.arch
            )));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Registry {
    pub root: PathBuf,
}

/// A model resolved for serving: either a registry entry (with metadata) or a
/// direct checkpoint directory on disk (metadata optional).
#[derive(Debug, Clone)]
pub enum ModelLocation {
    Registry {
        dir: PathBuf,
        meta: ModelMeta,
    },
    Direct {
        dir: PathBuf,
        meta: Option<ModelMeta>,
    },
}

impl ModelLocation {
    pub fn dir(&self) -> &Path {
        match self {
            ModelLocation::Registry { dir, .. } => dir,
            ModelLocation::Direct { dir, .. } => dir,
        }
    }

    pub fn meta(&self) -> Option<&ModelMeta> {
        match self {
            ModelLocation::Registry { meta, .. } => Some(meta),
            ModelLocation::Direct { meta, .. } => meta.as_ref(),
        }
    }

    pub fn display_name(&self) -> String {
        match self {
            ModelLocation::Registry { meta, .. } => meta.name.clone(),
            ModelLocation::Direct { dir, meta } => meta
                .as_ref()
                .map(|m| m.name.clone())
                .unwrap_or_else(|| dir.file_name().unwrap_or_default().to_string_lossy().into()),
        }
    }
}

/// Result of a registry listing: valid models plus entries that exist on disk
/// but fail validation (reported, never hidden).
#[derive(Debug, Default)]
pub struct RegistryListing {
    pub models: Vec<ModelMeta>,
    pub invalid: Vec<(String, String)>,
}

impl Registry {
    pub fn open(root: impl Into<PathBuf>) -> Registry {
        Registry { root: root.into() }
    }

    pub fn model_dir(&self, name: &str) -> Result<PathBuf, EdgeError> {
        validate_model_name(name)?;
        let dir = self.root.join(name);
        self.ensure_inside(&dir)?;
        Ok(dir)
    }

    /// Canonicalize and confirm the path stays inside the registry root.
    fn ensure_inside(&self, path: &Path) -> Result<(), EdgeError> {
        let root = fs::canonicalize(&self.root).map_err(|_| {
            EdgeError::not_found(format!(
                "registry root {} does not exist",
                self.root.display()
            ))
        })?;
        let target = fs::canonicalize(path)
            .map_err(|e| EdgeError::NotFound(format!("{}: {e}", path.display())))?;
        if !target.starts_with(&root) {
            return Err(EdgeError::invalid(format!(
                "path {} escapes the registry root {}",
                path.display(),
                self.root.display()
            )));
        }
        Ok(())
    }

    // -- import ------------------------------------------------------------

    /// Import a checkpoint directory into the registry. Copies the files,
    /// records SHA-256 checksums, derives the architecture from config.json.
    #[allow(clippy::too_many_arguments)]
    pub fn import(
        &self,
        source: &Path,
        name: &str,
        chat_template: ChatTemplate,
        quant_hint: &str,
        overwrite: bool,
        now_rfc3339: String,
    ) -> Result<ModelMeta, EdgeError> {
        validate_model_name(name)?;
        if !source.is_dir() {
            return Err(EdgeError::not_found(format!(
                "model source {} is not a directory",
                source.display()
            )));
        }
        let source = fs::canonicalize(source)
            .map_err(|e| EdgeError::not_found(format!("{}: {e}", source.display())))?;

        // Validate before anything is written: the checkpoint must load-shape.
        let config_path = source.join("config.json");
        let config: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&config_path).map_err(|e| {
                EdgeError::invalid(format!(
                    "not a nimbus checkpoint: reading {}: {e}",
                    config_path.display()
                ))
            })?)
            .map_err(|e| EdgeError::invalid(format!("config.json is not valid JSON: {e}")))?;
        for field in [
            "vocab_size",
            "n_layers",
            "n_heads",
            "d_model",
            "max_seq_len",
        ] {
            if config.get(field).and_then(|v| v.as_u64()).is_none() {
                return Err(EdgeError::invalid(format!(
                    "config.json is missing the integer field {field:?}"
                )));
            }
        }
        // Hand the config to the engine's parser: it rejects unknown norms,
        // activations and head-count mismatches.
        nimbus_infer_rs::engine::load_config(&config_path)
            .map_err(|e| EdgeError::invalid(format!("config.json rejected: {e}")))?;
        if !["weights.safetensors", "model.safetensors"]
            .iter()
            .any(|f| source.join(f).exists())
        {
            return Err(EdgeError::invalid(format!(
                "not a nimbus checkpoint: no weights.safetensors in {}",
                source.display()
            )));
        }
        // Tokenizer is optional but, when present, must be complete.
        let tok_dir = source.join("tokenizer");
        let has_tokenizer = tok_dir.is_dir();
        if has_tokenizer {
            for f in ["vocab.json", "merges.txt"] {
                if !tok_dir.join(f).exists() {
                    return Err(EdgeError::invalid(format!(
                        "tokenizer/ exists but is missing {f}"
                    )));
                }
            }
        }

        // Destination must not exist (or must be explicitly replaced).
        let dest_root = &self.root;
        let dest = dest_root.join(name);
        if dest.exists() {
            if !overwrite {
                return Err(EdgeError::invalid(format!(
                    "model {name:?} already exists in the registry; pass --overwrite to replace it"
                )));
            }
            fs::remove_dir_all(&dest)
                .map_err(|e| EdgeError::Internal(format!("removing {}: {e}", dest.display())))?;
        }
        fs::create_dir_all(&dest)
            .map_err(|e| EdgeError::Internal(format!("creating {}: {e}", dest.display())))?;

        // Copy every regular file below `source`, preserving relative paths.
        let mut files = BTreeMap::new();
        let mut stack = vec![source.clone()];
        while let Some(dir) = stack.pop() {
            for entry in fs::read_dir(&dir)
                .map_err(|e| EdgeError::Internal(format!("reading {}: {e}", dir.display())))?
            {
                let entry = entry
                    .map_err(|e| EdgeError::Internal(format!("reading {}: {e}", dir.display())))?;
                let path = entry.path();
                let rel = path
                    .strip_prefix(&source)
                    .expect("walk stays under source")
                    .to_string_lossy()
                    .replace('\\', "/");
                if path.is_dir() {
                    stack.push(path);
                    continue;
                }
                if !path.is_file() {
                    continue; // sockets, symlinks to nowhere, ... are skipped
                }
                let meta = copy_with_checksum(&path, &dest.join(&rel))?;
                files.insert(rel, meta);
            }
        }

        let arch = infer_arch(&config);
        let meta = ModelMeta {
            format: FORMAT_TAG.to_string(),
            name: name.to_string(),
            created_at: now_rfc3339,
            source: source.display().to_string(),
            chat_template: chat_template.as_meta_str().to_string(),
            arch,
            quant_hint: quant_hint.to_string(),
            files,
        };
        let meta_text = serde_json::to_string_pretty(&meta).expect("ModelMeta serializes");
        fs::write(dest.join(META_FILE), meta_text + "\n")
            .map_err(|e| EdgeError::Internal(format!("writing {META_FILE}: {e}")))?;
        Ok(meta)
    }

    // -- listing / inspect / remove -----------------------------------------

    pub fn list(&self) -> Result<RegistryListing, EdgeError> {
        let mut listing = RegistryListing::default();
        if !self.root.exists() {
            return Ok(listing);
        }
        let mut names: Vec<String> = fs::read_dir(&self.root)
            .map_err(|e| EdgeError::Internal(format!("reading {}: {e}", self.root.display())))?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().is_dir())
            .map(|e| e.file_name().to_string_lossy().into())
            .collect();
        names.sort();
        for name in names {
            if validate_model_name(&name).is_err() {
                listing
                    .invalid
                    .push((name, "invalid model name".to_string()));
                continue;
            }
            match self.load_meta(&name) {
                Ok(meta) => listing.models.push(meta),
                Err(e) => listing.invalid.push((name, e.to_string())),
            }
        }
        Ok(listing)
    }

    pub fn load_meta(&self, name: &str) -> Result<ModelMeta, EdgeError> {
        let dir = self.model_dir(name)?;
        let path = dir.join(META_FILE);
        let text = fs::read_to_string(&path).map_err(|_| {
            EdgeError::not_found(format!(
                "model {name:?} is not in the registry (missing {META_FILE} in {})",
                dir.display()
            ))
        })?;
        let meta: ModelMeta = serde_json::from_str(&text)
            .map_err(|e| EdgeError::invalid(format!("invalid {META_FILE} for {name:?}: {e}")))?;
        meta.validate(name)?;
        Ok(meta)
    }

    /// Inspect without loading weights: metadata + config summary + the
    /// safetensors header (tensor names and shapes only).
    pub fn inspect(&self, name: &str) -> Result<InspectReport, EdgeError> {
        let dir = self.model_dir(name)?;
        let meta = self.load_meta(name)?;
        let config_path = dir.join("config.json");
        let config = nimbus_infer_rs::engine::load_config(&config_path)
            .map_err(|e| EdgeError::invalid(format!("config.json rejected: {e}")))?;
        let weights = ["weights.safetensors", "model.safetensors"]
            .iter()
            .map(|f| dir.join(f))
            .find(|p| p.exists())
            .ok_or_else(|| EdgeError::invalid(format!("no weights file in {}", dir.display())))?;
        let tensors = read_safetensors_header(&weights)?;
        let mut tensor_names: Vec<String> = tensors.keys().cloned().collect();
        tensor_names.sort();
        let n_params: u64 = tensors
            .values()
            .map(|t| t.shape.iter().product::<usize>() as u64)
            .sum();
        Ok(InspectReport {
            meta,
            config: summarize(&config),
            weights_file: weights
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .into(),
            n_params,
            tensor_names,
            has_tokenizer: dir.join("tokenizer").is_dir(),
        })
    }

    pub fn remove(&self, name: &str) -> Result<(), EdgeError> {
        let dir = self.model_dir(name)?;
        self.load_meta(name)?; // refuse to delete things that are not registry models
        self.ensure_inside(&dir)?;
        fs::remove_dir_all(&dir)
            .map_err(|e| EdgeError::Internal(format!("removing {}: {e}", dir.display())))?;
        Ok(())
    }

    // -- resolve -------------------------------------------------------------

    /// Resolve `--model <name-or-path>`: a path that exists wins (serve from
    /// an arbitrary local directory, metadata optional); otherwise the string
    /// is looked up as a registry model.
    pub fn resolve(&self, name_or_path: &str) -> Result<ModelLocation, EdgeError> {
        let candidate = PathBuf::from(name_or_path);
        if candidate.is_dir() {
            let dir = fs::canonicalize(&candidate)
                .map_err(|e| EdgeError::not_found(format!("{}: {e}", candidate.display())))?;
            if !dir.join("config.json").exists() {
                return Err(EdgeError::not_found(format!(
                    "{} is a directory but has no config.json; not a nimbus checkpoint",
                    dir.display()
                )));
            }
            let meta = self.read_sidecar_meta(&dir)?;
            return Ok(ModelLocation::Direct { dir, meta });
        }
        match self.load_meta(name_or_path) {
            Ok(meta) => Ok(ModelLocation::Registry {
                dir: self.model_dir(name_or_path)?,
                meta,
            }),
            Err(EdgeError::NotFound(_)) => Err(EdgeError::not_found(format!(
                "model {name_or_path:?} not found: not a directory on disk and not a registry \
                 model (see `nimbus-edge models list`)"
            ))),
            Err(e) => Err(e),
        }
    }

    /// Read `edge-meta.json` from a direct (non-registry) model directory if
    /// present; missing file is fine, invalid file is an error.
    fn read_sidecar_meta(&self, dir: &Path) -> Result<Option<ModelMeta>, EdgeError> {
        let path = dir.join(META_FILE);
        if !path.exists() {
            return Ok(None);
        }
        let text = fs::read_to_string(&path)
            .map_err(|e| EdgeError::invalid(format!("reading {}: {e}", path.display())))?;
        let meta: ModelMeta = serde_json::from_str(&text)
            .map_err(|e| EdgeError::invalid(format!("invalid {META_FILE}: {e}")))?;
        // A sidecar may carry a different `name` than the directory (it is
        // informational for direct paths); only the format tag is enforced.
        if meta.format != FORMAT_TAG {
            return Err(EdgeError::invalid(format!(
                "unsupported model metadata format {:?} (expected {FORMAT_TAG:?})",
                meta.format
            )));
        }
        Ok(Some(meta))
    }
}

fn copy_with_checksum(src: &Path, dst: &Path) -> Result<FileMeta, EdgeError> {
    if let Some(parent) = dst.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| EdgeError::Internal(format!("creating {}: {e}", parent.display())))?;
    }
    let mut src_file = fs::File::open(src)
        .map_err(|e| EdgeError::Internal(format!("opening {}: {e}", src.display())))?;
    let mut dst_file = fs::File::create(dst)
        .map_err(|e| EdgeError::Internal(format!("creating {}: {e}", dst.display())))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; COPY_BUFFER_BYTES];
    let mut total = 0u64;
    loop {
        let n = src_file
            .read(&mut buf)
            .map_err(|e| EdgeError::Internal(format!("reading {}: {e}", src.display())))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
        dst_file
            .write_all(&buf[..n])
            .map_err(|e| EdgeError::Internal(format!("writing {}: {e}", dst.display())))?;
        total += n as u64;
    }
    Ok(FileMeta {
        sha256: hex(&hasher.finalize()),
        bytes: total,
    })
}

fn hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push_str(&format!("{b:02x}"));
    }
    out
}

/// Verify the on-disk files against the recorded checksums.
pub fn verify_checksums(dir: &Path, meta: &ModelMeta) -> Result<(), EdgeError> {
    for (rel, expect) in &meta.files {
        let path = dir.join(rel);
        let mut file = fs::File::open(&path).map_err(|_| {
            EdgeError::invalid(format!(
                "recorded file {rel:?} is missing from {}",
                dir.display()
            ))
        })?;
        let mut hasher = Sha256::new();
        std::io::copy(&mut file, &mut hasher)
            .map_err(|e| EdgeError::Internal(format!("hashing {}: {e}", path.display())))?;
        let got = hex(&hasher.finalize());
        if got != expect.sha256 {
            return Err(EdgeError::invalid(format!(
                "checksum mismatch for {rel:?}: file was modified after import"
            )));
        }
    }
    Ok(())
}

fn infer_arch(config: &serde_json::Value) -> String {
    match config.get("pos_encoding").and_then(|v| v.as_str()) {
        Some("rope") => "llama".to_string(),
        _ => "gpt2".to_string(),
    }
}

/// Model names: 1..=64 chars of [A-Za-z0-9._-], starting alphanumeric.
/// This is the path-traversal defense for registry lookups: no separators,
/// no leading dot, no "..".
pub fn validate_model_name(name: &str) -> Result<(), EdgeError> {
    let ok_len = !name.is_empty() && name.len() <= 64;
    let ok_chars = name
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '.' || c == '_' || c == '-');
    let ok_start = name
        .chars()
        .next()
        .is_some_and(|c| c.is_ascii_alphanumeric());
    if ok_len && ok_chars && ok_start {
        Ok(())
    } else {
        Err(EdgeError::invalid(format!(
            "invalid model name {name:?}: use 1..=64 characters of [A-Za-z0-9._-] starting \
             with an alphanumeric"
        )))
    }
}

// ---------------------------------------------------------------------------
// Safetensors header (metadata-only reading)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
pub struct TensorHeader {
    pub dtype: String,
    pub shape: Vec<usize>,
}

/// Read only the safetensors header: `u64 LE length` + JSON. Inspecting a
/// multi-GB model must not load its weights.
pub fn read_safetensors_header(path: &Path) -> Result<BTreeMap<String, TensorHeader>, EdgeError> {
    let mut file = fs::File::open(path)
        .map_err(|e| EdgeError::Internal(format!("opening {}: {e}", path.display())))?;
    let mut len_buf = [0u8; 8];
    file.read_exact(&mut len_buf).map_err(|e| {
        EdgeError::invalid(format!(
            "{}: too small for a safetensors header ({e})",
            path.display()
        ))
    })?;
    let header_len = u64::from_le_bytes(len_buf);
    if header_len == 0 || header_len > MAX_HEADER_BYTES {
        return Err(EdgeError::invalid(format!(
            "{}: implausible safetensors header length {header_len}",
            path.display()
        )));
    }
    let mut header_buf = vec![0u8; header_len as usize];
    file.read_exact(&mut header_buf).map_err(|e| {
        EdgeError::invalid(format!(
            "{}: truncated safetensors header ({e})",
            path.display()
        ))
    })?;
    let header: serde_json::Value = serde_json::from_slice(&header_buf).map_err(|e| {
        EdgeError::invalid(format!(
            "{}: safetensors header is not JSON: {e}",
            path.display()
        ))
    })?;
    let obj = header.as_object().ok_or_else(|| {
        EdgeError::invalid(format!(
            "{}: safetensors header is not an object",
            path.display()
        ))
    })?;
    let mut out = BTreeMap::new();
    for (name, info) in obj {
        if name == "__metadata__" {
            continue;
        }
        let dtype = info
            .get("dtype")
            .and_then(|v| v.as_str())
            .ok_or_else(|| EdgeError::invalid(format!("tensor {name:?}: missing dtype")))?;
        let shape = info
            .get("shape")
            .and_then(|v| v.as_array())
            .ok_or_else(|| EdgeError::invalid(format!("tensor {name:?}: missing shape")))?
            .iter()
            .map(|v| {
                v.as_u64()
                    .map(|n| n as usize)
                    .ok_or_else(|| EdgeError::invalid(format!("tensor {name:?}: bad shape")))
            })
            .collect::<Result<Vec<usize>, EdgeError>>()?;
        out.insert(
            name.clone(),
            TensorHeader {
                dtype: dtype.to_string(),
                shape,
            },
        );
    }
    if out.is_empty() {
        return Err(EdgeError::invalid(format!(
            "{}: safetensors header lists no tensors",
            path.display()
        )));
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Inspect report
// ---------------------------------------------------------------------------

/// A serializable summary of `nimbus_infer_rs::config::ModelConfig` (that
/// type belongs to the engine crate and carries no serde impls).
#[derive(Debug, Serialize)]
pub struct ConfigSummary {
    pub name: String,
    pub vocab_size: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub d_model: usize,
    pub d_mlp: usize,
    pub max_seq_len: usize,
    pub norm: String,
    pub activation: String,
    pub pos_encoding: String,
    pub tie_embeddings: bool,
}

fn summarize(config: &nimbus_infer_rs::config::ModelConfig) -> ConfigSummary {
    use nimbus_infer_rs::config::{Activation, Norm, PosEncoding};
    ConfigSummary {
        name: config.name.clone(),
        vocab_size: config.vocab_size,
        n_layers: config.n_layers,
        n_heads: config.n_heads,
        d_model: config.d_model,
        d_mlp: config.d_mlp,
        max_seq_len: config.max_seq_len,
        norm: match config.norm {
            Norm::LayerNorm => "layernorm",
            Norm::RmsNorm => "rmsnorm",
        }
        .to_string(),
        activation: match config.activation {
            Activation::Gelu => "gelu",
            Activation::Swiglu => "swiglu",
        }
        .to_string(),
        pos_encoding: match config.pos_encoding {
            PosEncoding::Learned => "learned",
            PosEncoding::Rope => "rope",
        }
        .to_string(),
        tie_embeddings: config.tie_embeddings,
    }
}

#[derive(Debug, Serialize)]
pub struct InspectReport {
    pub meta: ModelMeta,
    pub config: ConfigSummary,
    pub weights_file: String,
    pub n_params: u64,
    pub tensor_names: Vec<String>,
    pub has_tokenizer: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_name_validation_blocks_traversal() {
        for ok in ["tiny-gpt2", "gpt2", "SmolLM2_135M", "a.b-c_d", "x1"] {
            assert!(validate_model_name(ok).is_ok(), "{ok} should be valid");
        }
        for bad in [
            "",
            ".",
            "..",
            "../etc",
            "a/b",
            "a\\b",
            ".hidden",
            "-x",
            "a b",
            "模型",
            "way-too-long-__________________________________________________________________x",
        ] {
            assert!(validate_model_name(bad).is_err(), "{bad} should be invalid");
        }
        // The classic traversal payloads must all fail.
        assert!(validate_model_name("..%2f..").is_err());
        // Dots inside a name are fine; separators are not.
        assert!(validate_model_name("a..b").is_ok());
    }

    #[test]
    fn meta_rejects_unknown_fields_and_wrong_format() {
        let text = r#"{"format":"nimbus-checkpoint-v1","name":"m","created_at":"2026-01-01T00:00:00Z",
            "source":"/x","chat_template":"none","arch":"gpt2","quant_hint":"fp32","files":{},"extra":1}"#;
        assert!(serde_json::from_str::<ModelMeta>(text).is_err());
        let text = text.replace(",\"extra\":1}", "}");
        assert!(serde_json::from_str::<ModelMeta>(&text).is_ok());
        let wrong_format = text.replace(FORMAT_TAG, "some-other-format");
        assert!(serde_json::from_str::<ModelMeta>(&wrong_format).is_ok()); // shape is fine
        let meta: ModelMeta = serde_json::from_str(&wrong_format).unwrap();
        assert!(meta.validate("m").is_err()); // but validation rejects the format tag
    }

    #[test]
    fn header_reader_rejects_truncated_and_garbage() {
        let dir = std::env::temp_dir().join("nb-registry-test-header");
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("bad.safetensors");
        fs::write(&path, [0u8; 3]).unwrap();
        assert!(read_safetensors_header(&path).is_err());
        fs::write(&path, 64u64.to_le_bytes()).unwrap();
        assert!(read_safetensors_header(&path).is_err());
        fs::write(&path, 8u64.to_le_bytes()).unwrap();
        assert!(read_safetensors_header(&path).is_err());
        let _ = fs::remove_dir_all(&dir);
    }
}
