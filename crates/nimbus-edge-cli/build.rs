// Capture build metadata (rustc version, target, profile) for `bench` and
// `status` output. Plain cargo-provided env vars plus one rustc invocation.
use std::process::Command;

fn main() {
    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".to_string());
    let version = Command::new(&rustc)
        .arg("--version")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".to_string());
    println!("cargo:rustc-env=NIMBUS_BUILD_RUSTC={version}");

    let target = std::env::var("TARGET").unwrap_or_else(|_| "unknown".to_string());
    println!("cargo:rustc-env=NIMBUS_BUILD_TARGET={target}");

    let profile = std::env::var("PROFILE").unwrap_or_else(|_| "debug".to_string());
    println!("cargo:rustc-env=NIMBUS_BUILD_PROFILE={profile}");

    println!("cargo:rerun-if-changed=build.rs");
}
