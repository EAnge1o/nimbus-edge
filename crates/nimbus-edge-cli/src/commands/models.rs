//! `nimbus-edge models`: list | import | inspect | remove.

use std::path::PathBuf;

use clap::Subcommand;

use nimbus_edge_core::chat::ChatTemplate;
use nimbus_edge_core::error::EdgeError;
use nimbus_edge_core::registry::Registry;

#[derive(Subcommand, Debug)]
pub enum ModelsCommand {
    /// List models in the registry (including entries that fail validation).
    List {
        #[arg(long)]
        json: bool,
    },
    /// Import a checkpoint directory into the registry.
    Import {
        /// Checkpoint directory (config.json + weights.safetensors [+ tokenizer/]).
        #[arg(long)]
        path: PathBuf,
        /// Registry name (1..=64 chars of [A-Za-z0-9._-], starts alphanumeric).
        #[arg(long)]
        name: String,
        /// Chat template for /v1/chat/completions ("none" keeps the model
        /// completions-only).
        #[arg(long, default_value = "none")]
        chat_template: String,
        /// Quantization hint recorded in the metadata (serve-time --quant
        /// overrides this at load).
        #[arg(long, default_value = "fp32")]
        quant: String,
        #[arg(long)]
        overwrite: bool,
    },
    /// Show metadata, config summary and tensor list without loading weights.
    Inspect {
        #[arg()]
        name: String,
    },
    /// Remove a model from the registry.
    Remove {
        #[arg()]
        name: String,
    },
}

pub fn run(models_dir: &std::path::Path, cmd: ModelsCommand) -> Result<(), EdgeError> {
    let registry = Registry::open(models_dir);
    match cmd {
        ModelsCommand::List { json } => {
            let listing = registry.list()?;
            if json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&serde_json::json!({
                        "models_dir": models_dir.display().to_string(),
                        "models": listing.models,
                        "invalid": listing.invalid.iter().map(|(n, e)| serde_json::json!({
                            "name": n, "error": e,
                        })).collect::<Vec<_>>(),
                    }))
                    .expect("listing serializes")
                );
                return Ok(());
            }
            println!("models dir: {}", models_dir.display());
            if listing.models.is_empty() && listing.invalid.is_empty() {
                println!("(empty — import one with: nimbus-edge models import --path <dir> --name <name>)");
                return Ok(());
            }
            println!(
                "{:<20} {:>6} {:>10} {:>6}  {:<8} {:<10}",
                "NAME", "ARCH", "SIZE", "PARAMS", "QUANT", "TEMPLATE"
            );
            for m in &listing.models {
                println!(
                    "{:<20} {:>6} {:>10} {:>6}  {:<8} {:<10}",
                    m.name,
                    m.arch,
                    human_size(m.total_bytes()),
                    human_params(m),
                    m.quant_hint,
                    m.chat_template,
                );
            }
            for (name, err) in &listing.invalid {
                println!("{:<20} INVALID: {err}", name);
            }
            Ok(())
        }
        ModelsCommand::Import {
            path,
            name,
            chat_template,
            quant,
            overwrite,
        } => {
            let template = ChatTemplate::parse(&chat_template)?;
            let meta = registry.import(&path, &name, template, &quant, overwrite, rfc3339_now())?;
            println!(
                "imported {} ({} files, {}, template: {})",
                meta.name,
                meta.files.len(),
                human_size(meta.total_bytes()),
                meta.chat_template,
            );
            Ok(())
        }
        ModelsCommand::Inspect { name } => {
            let report = registry.inspect(&name)?;
            println!(
                "{}",
                serde_json::to_string_pretty(&report).expect("inspect report serializes")
            );
            Ok(())
        }
        ModelsCommand::Remove { name } => {
            registry.remove(&name)?;
            println!("removed {name}");
            Ok(())
        }
    }
}

fn human_params(m: &nimbus_edge_core::registry::ModelMeta) -> String {
    let bytes = m
        .files
        .get("weights.safetensors")
        .or_else(|| m.files.get("model.safetensors"))
        .map(|f| f.bytes)
        .unwrap_or(0);
    if bytes == 0 {
        return "-".into();
    }
    // fp32 dominant: params ≈ bytes/4 (int8/int4 hints adjust the label only).
    let approx = bytes / 4;
    if approx >= 1_000_000 {
        format!("{:.1}M", approx as f64 / 1_000_000.0)
    } else if approx >= 1_000 {
        format!("{:.0}K", approx as f64 / 1_000.0)
    } else {
        format!("{approx}")
    }
}

fn human_size(v: u64) -> String {
    if v >= 1 << 30 {
        format!("{:.1}GB", v as f64 / (1 << 30) as f64)
    } else if v >= 1 << 20 {
        format!("{:.1}MB", v as f64 / (1 << 20) as f64)
    } else if v >= 1 << 10 {
        format!("{:.1}KB", v as f64 / (1 << 10) as f64)
    } else {
        format!("{v}B")
    }
}

/// Local time as RFC 3339 UTC, from the epoch seconds we can get without a
/// date library (format is honest UTC, no timezone suffix games).
fn rfc3339_now() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let days = secs / 86_400;
    let rem = secs % 86_400;
    let (h, mi, s) = (rem / 3600, (rem % 3600) / 60, rem % 60);
    // civil-from-days (Howard Hinnant's algorithm).
    let z = days as i64 + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}-{m:02}-{d:02}T{h:02}:{mi:02}:{s:02}Z")
}
