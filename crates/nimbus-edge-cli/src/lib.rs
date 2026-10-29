//! nimbus-edge CLI: `serve`, `models`, `status`, `bench`.
//!
//! `run()` parses argv and dispatches; each command lives in `commands/`.
//! The lib/bin split lets the black-box integration test exercise the exact
//! binary a release ships.

use clap::{Parser, Subcommand};

pub mod commands;
pub mod platform;

#[derive(Parser)]
#[command(
    name = "nimbus-edge",
    version = nimbus_edge_core::VERSION,
    about = "A local-first, single-binary LLM inference server (OpenAI-compatible subset)"
)]
pub struct Cli {
    /// Model registry directory (default: ~/.nimbus-edge/models).
    #[arg(long, global = true, env = "NIMBUS_EDGE_MODELS_DIR")]
    pub models_dir: Option<std::path::PathBuf>,

    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand)]
pub enum Command {
    /// Start the OpenAI-compatible HTTP server.
    Serve(commands::serve::ServeArgs),
    /// Manage the local model registry.
    #[command(subcommand)]
    Models(commands::models::ModelsCommand),
    /// Show local build/registry info, or probe a running server.
    Status(commands::status::StatusArgs),
    /// Measure prefill/decode throughput and first-token latency.
    Bench(commands::bench::BenchArgs),
}

/// Default registry directory: `<home>/.nimbus-edge/models`.
pub fn default_models_dir() -> std::path::PathBuf {
    let home = std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .unwrap_or_else(|_| ".".to_string());
    std::path::PathBuf::from(home)
        .join(".nimbus-edge")
        .join("models")
}

pub fn run() -> i32 {
    let cli = Cli::parse();
    let models_dir = cli.models_dir.unwrap_or_else(default_models_dir);
    let code = match cli.command {
        Command::Serve(args) => commands::serve::run(&models_dir, args),
        Command::Models(cmd) => commands::models::run(&models_dir, cmd),
        Command::Status(args) => commands::status::run(&models_dir, args),
        Command::Bench(args) => commands::bench::run(&models_dir, args),
    };
    match code {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("error: {e}");
            1
        }
    }
}
