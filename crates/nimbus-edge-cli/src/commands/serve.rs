//! `nimbus-edge serve`: one command from model to working HTTP endpoint.

use std::sync::Arc;

use clap::Args;

use nimbus_edge_core::backend::{Backend, NimbusBackend, PrefixCacheConfig};
use nimbus_edge_core::chat::ChatTemplate;
use nimbus_edge_core::error::EdgeError;
use nimbus_edge_core::registry::Registry;
use nimbus_edge_core::scheduler::{Scheduler, SchedulerConfig};
use nimbus_edge_server::ServerConfig;

#[derive(Args, Debug)]
pub struct ServeArgs {
    /// Registry model name, or a path to a checkpoint directory.
    #[arg(long)]
    pub model: String,

    #[arg(long, default_value = "127.0.0.1")]
    pub host: String,

    /// 0 picks a free port; the actual address is printed on startup.
    #[arg(long, default_value_t = 11434)]
    pub port: u16,

    /// Load-time weight quantization.
    #[arg(long, default_value = "fp32")]
    pub quant: String,

    /// Generations running at once (each has its own KV cache; 1 keeps
    /// decode memory-bandwidth friendly).
    #[arg(long, default_value_t = 1)]
    pub max_concurrency: usize,

    /// Requests allowed to wait when all concurrency slots are busy; a full
    /// queue rejects with HTTP 429.
    #[arg(long, default_value_t = 32)]
    pub max_queue_size: usize,

    /// Require this bearer token on /v1/* (also: NIMBUS_EDGE_AUTH_TOKEN).
    #[arg(long, env = "NIMBUS_EDGE_AUTH_TOKEN", hide_env_values = true)]
    pub auth_token: Option<String>,

    /// Threads for the engine's internal kernels (default: all cores;
    /// decode caps itself at 4 for memory-bandwidth reasons).
    #[arg(long)]
    pub threads: Option<usize>,

    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    pub prefix_cache: bool,

    #[arg(long, default_value_t = 2)]
    pub prefix_cache_entries: usize,

    /// Allow browser clients from these origins (comma-separated, or "*" for
    /// any). Off by default — see SECURITY.md.
    #[arg(long)]
    pub cors_origins: Option<String>,
}

pub fn run(models_dir: &std::path::Path, args: ServeArgs) -> Result<(), EdgeError> {
    init_tracing();
    if let Some(n) = args.threads {
        if n == 0 {
            return Err(EdgeError::invalid("--threads must be >= 1"));
        }
        std::env::set_var("NIMBUS_THREADS", n.to_string());
    }
    let quant =
        nimbus_infer_rs::engine::QuantMode::parse(&args.quant).map_err(EdgeError::invalid)?;

    let registry = Registry::open(models_dir);
    let location = registry.resolve(&args.model)?;
    let dir = location.dir().to_path_buf();
    let display_name = location.display_name();
    let template = match location.meta() {
        Some(meta) => meta.chat_template()?,
        None => {
            eprintln!(
                "note: {dir:?} has no edge-meta.json; chat template defaults to none \
                 (chat completions will fail until the model is imported with --chat-template)"
            );
            ChatTemplate::None
        }
    };

    eprintln!("loading model {display_name} ({}) ...", args.quant);
    let prefix_cfg = PrefixCacheConfig {
        enabled: args.prefix_cache,
        max_entries: args.prefix_cache_entries,
        ..Default::default()
    };
    let backend = NimbusBackend::load(&dir, quant, display_name.clone(), template, prefix_cfg)?;
    let info = backend.info();

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|e| EdgeError::Internal(format!("tokio runtime: {e}")))?;
    rt.block_on(async move {
        // The dispatcher task needs a runtime; spawn it inside block_on.
        let scheduler = Scheduler::spawn(
            Arc::new(backend),
            SchedulerConfig {
                max_queue_size: args.max_queue_size,
                max_concurrency: args.max_concurrency,
            },
        );
        let addr = format!("{}:{}", args.host, args.port);
        let listener = tokio::net::TcpListener::bind(&addr)
            .await
            .map_err(|e| EdgeError::Internal(format!("binding {addr}: {e}")))?;
        let local = listener
            .local_addr()
            .map_err(|e| EdgeError::Internal(format!("local addr: {e}")))?;
        let base = format!("http://{}", local);

        print_banner(&base, &info, &args);
        let cors_origins = args.cors_origins.as_ref().map(|list| {
            list.split(',')
                .map(|o| o.trim().to_string())
                .filter(|o| !o.is_empty())
                .collect::<Vec<_>>()
        });
        let cfg = ServerConfig {
            served_model_name: info.name.clone(),
            auth_token: args.auth_token.clone(),
            cors_origins,
        };
        nimbus_edge_server::serve(listener, scheduler, info, cfg)
            .await
            .map_err(|e| EdgeError::Internal(format!("server: {e}")))
    })
}

fn init_tracing() {
    use tracing_subscriber::EnvFilter;
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    let _ = tracing_subscriber::fmt().with_env_filter(filter).try_init();
}

fn print_banner(base: &str, info: &nimbus_edge_core::backend::ModelInfo, args: &ServeArgs) {
    let quant_note = if info.kv_cache_bytes > 0 {
        format!(
            ", weights {:.1} MB, kv cache {:.1} MB/request",
            info.weight_bytes as f64 / (1024.0 * 1024.0),
            info.kv_cache_bytes as f64 / (1024.0 * 1024.0)
        )
    } else {
        String::new()
    };
    let model = &info.name;
    println!(
        "nimbus-edge v{version} | model {model} ({arch}, {quant}{quant_note}) | context {ctx} tokens",
        version = nimbus_edge_core::VERSION,
        arch = info.arch,
        quant = info.quant,
        ctx = info.context_len,
    );
    println!(
        "listening on {base} | max_concurrency {cc} | max_queue_size {qs} | auth {}",
        if args.auth_token.is_some() {
            "on"
        } else {
            "off"
        },
        cc = args.max_concurrency,
        qs = args.max_queue_size,
    );
    println!();
    println!("Try:");
    println!(r#"  curl {base}/v1/chat/completions -H "Content-Type: application/json" \"#);
    println!(
        r#"       -d '{{"model":"{model}","messages":[{{"role":"user","content":"hello"}}]}}'"#
    );
}
