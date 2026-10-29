//! `nimbus-edge status`: local build/registry facts; with --url, probe a
//! running server's /healthz.

use clap::Args;

use nimbus_edge_core::error::EdgeError;
use nimbus_edge_core::registry::Registry;

#[derive(Args, Debug)]
pub struct StatusArgs {
    /// Probe a running server's /healthz instead of printing local info.
    #[arg(long)]
    pub url: Option<String>,
}

pub fn run(models_dir: &std::path::Path, args: StatusArgs) -> Result<(), EdgeError> {
    if let Some(url) = args.url {
        return probe(&url);
    }

    let registry = Registry::open(models_dir);
    let listing = registry.list()?;
    let build = crate::platform::build_info();
    println!("nimbus-edge v{}", nimbus_edge_core::VERSION);
    println!(
        "build:     rustc {}, target {}, profile {}",
        build["rustc"], build["target"], build["profile"]
    );
    println!(
        "platform:  {} {}",
        std::env::consts::OS,
        std::env::consts::ARCH
    );
    if let Some(cpu) = crate::platform::cpu_model() {
        println!("cpu:       {cpu}");
    }
    if let Some(mem) = crate::platform::total_memory_bytes() {
        println!(
            "memory:    {:.1} GB total",
            mem as f64 / (1024.0 * 1024.0 * 1024.0)
        );
    }
    println!("models:    {}", models_dir.display());
    println!(
        "           {} registered, {} invalid",
        listing.models.len(),
        listing.invalid.len()
    );
    for m in &listing.models {
        println!(
            "           - {} ({}, template {})",
            m.name, m.arch, m.chat_template
        );
    }
    Ok(())
}

fn probe(url: &str) -> Result<(), EdgeError> {
    let base = url.trim_end_matches('/');
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| EdgeError::Internal(format!("tokio runtime: {e}")))?;
    rt.block_on(async {
        let response = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            reqwest::get(format!("{base}/healthz")),
        )
        .await
        .map_err(|_| EdgeError::not_found(format!("{base}/healthz: timed out after 5s")))?
        .map_err(|e| EdgeError::not_found(format!("{base}/healthz: {e}")))?;
        let status = response.status();
        let body: serde_json::Value = response
            .json()
            .await
            .map_err(|e| EdgeError::Internal(format!("healthz body: {e}")))?;
        println!("{base}");
        println!(
            "{}",
            serde_json::to_string_pretty(&body).unwrap_or_default()
        );
        if !status.is_success() {
            return Err(EdgeError::Internal(format!("healthz returned {status}")));
        }
        Ok(())
    })
}
