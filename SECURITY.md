# Security

## Exposure model

Nimbus Edge is a **local development server**. It binds `127.0.0.1` by
default and is meant to serve one user on one machine. Exposing it beyond
localhost (`--host 0.0.0.0`) is an explicit operator decision; if you do,
set a bearer token (`--auth-token` or `NIMBUS_EDGE_AUTH_TOKEN`) and use a
reverse proxy with TLS in front of it. The server has no TLS of its own.

## Auth

- When a token is configured, every `/v1/*` request requires
  `Authorization: Bearer <token>`; failures get a 401 with an OpenAI-style
  error body. Token comparison is constant-time.
- `/healthz` deliberately stays open (no model text, only counters) so
  liveness probes work without credentials.

## Input handling

- **Strict request parsing**: unknown JSON fields and known-but-unimplemented
  parameters are rejected with 400s; nothing is silently ignored or guessed.
- **Path traversal**: model names are restricted to
  `[A-Za-z0-9._-]` (1–64 chars, starting alphanumeric). Every registry
  access canonicalizes the path and requires it to stay inside the registry
  root. Import copies files; it does not follow user-supplied symlinks into
  the registry (non-regular files are skipped).
- **Metadata integrity**: `edge-meta.json` uses `deny_unknown_fields` and is
  validated on every load; per-file SHA-256 checksums recorded at import let
  you detect post-import tampering (`verify_checksums`).
- **Safetensors robustness**: headers are size-capped and tensor data
  offsets are bounds-checked against the file size (a truncated or corrupt
  file is a clean error, not a crash).
- **Resource bounds**: bounded request queue (429 when full), bounded
  concurrency, bounded SSE channel with backpressure, prompt-length checks
  against the model context, `max_tokens` ceiling. No unbounded allocation
  is driven by request size alone.
- **Prompt content**: prompts are tokenized and run through the model; the
  server never evaluates or stores them beyond the lifetime of the request
  (plus the optional in-memory prefix cache, which lives and dies with the
  process).

## Known non-goals

- No multi-tenancy, no per-user quotas, no audit log.
- No sandboxing of the model directory: whoever can run `nimbus-edge serve`
  can read any local path they point it at — same trust level as any local
  CLI tool.
- The bearer token protects the API; it does not encrypt traffic (use a TLS
  proxy for that).

## Reporting

This is a small open-source project — please open a GitHub issue for
security problems, or use the contact information on the repository's
profile page for anything sensitive. Fixes land on `main` and are noted in
[CHANGELOG.md](CHANGELOG.md).
