//! nimbus-edge-core: the request model, chat templates, model registry,
//! scheduling and caching for Nimbus Edge.
//!
//! The crate boundary is deliberate: everything here is HTTP-agnostic. The
//! server crate maps OpenAI-compatible HTTP onto [`scheduler::Scheduler`];
//! the numeric kernel lives in `nimbus-infer-rs`, which this crate consumes
//! as a dependency through [`backend::NimbusBackend`].

pub mod backend;
pub mod chat;
pub mod error;
pub mod openai;
pub mod registry;
pub mod scheduler;

#[cfg(feature = "test-support")]
pub mod fixture;

/// Crate version, surfaced in `/healthz` and `nimbus-edge status`.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

const _: () = {
    // The engine is shared across request threads; the type system must
    // guarantee that. `Engine` only owns plain immutable data, but this pins
    // the property so a future interior-mutability regression fails to build.
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<nimbus_infer_rs::engine::Engine>();
};
