//! Request scheduler: a bounded FIFO queue feeding a bounded number of
//! in-flight generations.
//!
//! Contract:
//! - `max_queue_size` bounds *waiting* requests; a full queue rejects
//!   immediately with [`EdgeError::Busy`] (HTTP 429). Nothing queues
//!   unboundedly.
//! - `max_concurrency` bounds *running* generations. Each runs on its own
//!   blocking thread with its own engine-side KV cache, so concurrent
//!   requests never share sequence state.
//! - Cancellation works at every stage: dropping the [`SubmitHandle`]
//!   receivers marks the request dead (queued entries are swept without ever
//!   running; a running generation is stopped between decode steps via the
//!   engine's cancellation hook), and [`Scheduler::cancel`] does the same by
//!   request id.
//!
//! This is deliberately *not* continuous batching: requests do not share a
//! forward pass. It is a safe, honest bounded-concurrency queue.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use tokio::sync::{mpsc, oneshot, Notify};

use crate::backend::{Backend, CacheStats, FinishReason, GenOutput, GenRequest};
use crate::error::EdgeError;

/// Events streamed to a request's owner, in order.
#[derive(Debug)]
pub enum StreamEvent {
    /// Decoded text increment (never splitting a UTF-8 character).
    Token(String),
    /// Terminal event; sent exactly once.
    Done(Result<GenOutput, EdgeError>),
}

struct Job {
    id: u64,
    req: GenRequest,
    tx: mpsc::Sender<StreamEvent>,
    cancel: Arc<AtomicBool>,
    /// Closed when the request owner goes away (client disconnect).
    alive: oneshot::Sender<()>,
}

/// What a submitter keeps to observe its request. Dropping `rx` *or* `alive`
/// cancels the request: queued entries are removed, running ones stop.
pub struct SubmitHandle {
    pub id: u64,
    pub rx: mpsc::Receiver<StreamEvent>,
    /// Must live as long as the request is observed; see struct doc.
    pub alive: oneshot::Receiver<()>,
}

#[derive(Debug, Clone, Copy)]
pub struct SchedulerConfig {
    pub max_queue_size: usize,
    pub max_concurrency: usize,
}

#[derive(Debug, Clone, Copy, Default, serde::Serialize)]
pub struct SchedulerStats {
    pub queue_depth: usize,
    pub inflight: usize,
    pub max_queue_size: usize,
    pub max_concurrency: usize,
    pub submitted: u64,
    pub completed: u64,
    pub failed: u64,
    pub cancelled: u64,
    pub queue_full_rejections: u64,
}

pub struct Scheduler {
    backend: Arc<dyn Backend>,
    cfg: SchedulerConfig,
    queue: Mutex<VecDeque<Job>>,
    active_cancels: Mutex<HashMap<u64, Arc<AtomicBool>>>,
    notify: Notify,
    next_id: AtomicU64,
    inflight: AtomicU64,
    submitted: AtomicU64,
    completed: AtomicU64,
    failed: AtomicU64,
    cancelled: AtomicU64,
    queue_full: AtomicU64,
}

impl Scheduler {
    /// Create the scheduler and spawn its dispatcher task.
    pub fn spawn(backend: Arc<dyn Backend>, cfg: SchedulerConfig) -> Arc<Scheduler> {
        let scheduler = Arc::new(Scheduler {
            backend,
            cfg,
            queue: Mutex::new(VecDeque::new()),
            active_cancels: Mutex::new(HashMap::new()),
            notify: Notify::new(),
            next_id: AtomicU64::new(1),
            inflight: AtomicU64::new(0),
            submitted: AtomicU64::new(0),
            completed: AtomicU64::new(0),
            failed: AtomicU64::new(0),
            cancelled: AtomicU64::new(0),
            queue_full: AtomicU64::new(0),
        });
        tokio::spawn(scheduler.clone().dispatcher());
        scheduler
    }

    pub fn backend(&self) -> &Arc<dyn Backend> {
        &self.backend
    }

    /// Enqueue a request. Fails with [`EdgeError::Busy`] when the queue is
    /// full — the caller maps that to HTTP 429.
    pub fn submit(&self, req: GenRequest) -> Result<SubmitHandle, EdgeError> {
        self.sweep();
        let mut queue = self.queue.lock().unwrap();
        if queue.len() >= self.cfg.max_queue_size {
            self.queue_full.fetch_add(1, Ordering::Relaxed);
            return Err(EdgeError::Busy(format!(
                "request queue is full ({}/{} entries); retry later",
                queue.len(),
                self.cfg.max_queue_size
            )));
        }
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = mpsc::channel(256);
        let (alive, alive_rx) = oneshot::channel();
        queue.push_back(Job {
            id,
            req,
            tx,
            cancel: Arc::new(AtomicBool::new(false)),
            alive,
        });
        drop(queue);
        self.submitted.fetch_add(1, Ordering::Relaxed);
        self.notify.notify_one();
        Ok(SubmitHandle {
            id,
            rx,
            alive: alive_rx,
        })
    }

    /// Cancel by request id. Returns true if a queued entry was removed; a
    /// running generation is flagged to stop between decode steps either way.
    pub fn cancel(&self, id: u64) -> bool {
        let removed = {
            let mut queue = self.queue.lock().unwrap();
            let before = queue.len();
            queue.retain(|j| j.id != id);
            queue.len() != before
        };
        if let Some(flag) = self.active_cancels.lock().unwrap().get(&id) {
            flag.store(true, Ordering::Relaxed);
        }
        if removed {
            self.cancelled.fetch_add(1, Ordering::Relaxed);
        }
        removed
    }

    /// Cancel everything (server shutdown).
    pub fn shutdown(&self) {
        self.queue.lock().unwrap().clear();
        for flag in self.active_cancels.lock().unwrap().values() {
            flag.store(true, Ordering::Relaxed);
        }
    }

    /// Remove dead entries (their observer went away) without running them.
    fn sweep(&self) {
        let mut queue = self.queue.lock().unwrap();
        let before = queue.len();
        queue.retain(|j| !j.tx.is_closed());
        let removed = before - queue.len();
        if removed > 0 {
            tracing::debug!(removed, "swept cancelled queue entries");
        }
    }

    fn dispatch_ready(self: &Arc<Self>) {
        self.sweep();
        loop {
            if self.inflight.load(Ordering::SeqCst) >= self.cfg.max_concurrency as u64 {
                return;
            }
            let job = {
                let mut queue = self.queue.lock().unwrap();
                loop {
                    match queue.pop_front() {
                        Some(j) if j.tx.is_closed() => {
                            // Cancelled while queued: never runs.
                            self.cancelled.fetch_add(1, Ordering::Relaxed);
                            continue;
                        }
                        other => break other,
                    }
                }
            };
            let Some(job) = job else { return };
            self.inflight.fetch_add(1, Ordering::SeqCst);
            self.active_cancels
                .lock()
                .unwrap()
                .insert(job.id, job.cancel.clone());
            tokio::spawn(run_job(self.clone(), job));
        }
    }

    async fn dispatcher(self: Arc<Self>) {
        loop {
            self.dispatch_ready();
            self.notify.notified().await;
        }
    }

    pub fn stats(&self) -> SchedulerStats {
        SchedulerStats {
            queue_depth: self.queue.lock().unwrap().len(),
            inflight: self.inflight.load(Ordering::SeqCst) as usize,
            max_queue_size: self.cfg.max_queue_size,
            max_concurrency: self.cfg.max_concurrency,
            submitted: self.submitted.load(Ordering::Relaxed),
            completed: self.completed.load(Ordering::Relaxed),
            failed: self.failed.load(Ordering::Relaxed),
            cancelled: self.cancelled.load(Ordering::Relaxed),
            queue_full_rejections: self.queue_full.load(Ordering::Relaxed),
        }
    }

    pub fn cache_stats(&self) -> CacheStats {
        self.backend.cache_stats()
    }
}

async fn run_job(scheduler: Arc<Scheduler>, job: Job) {
    let backend = scheduler.backend.clone();
    let Job {
        id,
        req,
        tx,
        cancel,
        alive,
    } = job;
    let tx_out = tx.clone(); // survives the blocking closure for the Done event
    let result = tokio::task::spawn_blocking(move || {
        let is_cancelled = || cancel.load(Ordering::Relaxed) || alive.is_closed();
        let events = tx;
        backend.generate(&req, &is_cancelled, &mut |delta| {
            if events.blocking_send(StreamEvent::Token(delta)).is_err() {
                // The observer is gone: stop spending CPU on this request.
                cancel.store(true, Ordering::Relaxed);
            }
        })
    })
    .await;

    let out = match result {
        Ok(Ok(output)) => {
            if output.finish_reason == FinishReason::Cancelled || tx_out.is_closed() {
                scheduler.cancelled.fetch_add(1, Ordering::Relaxed);
            } else {
                scheduler.completed.fetch_add(1, Ordering::Relaxed);
            }
            Ok(output)
        }
        Ok(Err(e)) => {
            scheduler.failed.fetch_add(1, Ordering::Relaxed);
            Err(e)
        }
        Err(join) => {
            scheduler.failed.fetch_add(1, Ordering::Relaxed);
            Err(EdgeError::Internal(format!(
                "generation task panicked: {join}"
            )))
        }
    };
    let _ = tx_out.send(StreamEvent::Done(out)).await;

    scheduler.inflight.fetch_sub(1, Ordering::SeqCst);
    scheduler.active_cancels.lock().unwrap().remove(&id);
    scheduler.notify.notify_one();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::openai::GenParams;
    use std::time::Duration;

    /// A backend whose generation is slow enough to test queueing and
    /// cancellation deterministically, with no model on disk at all.
    struct StubBackend {
        per_token_delay: Duration,
    }

    impl Backend for StubBackend {
        fn info(&self) -> crate::backend::ModelInfo {
            unimplemented!("not needed by scheduler tests")
        }

        fn generate(
            &self,
            req: &GenRequest,
            cancelled: &dyn Fn() -> bool,
            on_delta: &mut dyn FnMut(String),
        ) -> Result<GenOutput, EdgeError> {
            for i in 0..req.params.max_new_tokens {
                if cancelled() {
                    return Ok(GenOutput {
                        text: String::new(),
                        finish_reason: FinishReason::Cancelled,
                        prompt_tokens: 1,
                        completion_tokens: i as usize,
                    });
                }
                std::thread::sleep(self.per_token_delay);
                on_delta(format!("t{i} "));
            }
            Ok(GenOutput {
                text: String::new(),
                finish_reason: FinishReason::Length,
                prompt_tokens: 1,
                completion_tokens: req.params.max_new_tokens as usize,
            })
        }

        fn cache_stats(&self) -> CacheStats {
            CacheStats::default()
        }
    }

    fn params(tokens: u32) -> GenRequest {
        GenRequest {
            prompt: "p".into(),
            params: GenParams {
                max_new_tokens: tokens,
                temperature: 0.0,
                top_p: 1.0,
                seed: None,
                stop_strings: Vec::new(),
            },
        }
    }

    async fn spawn_stub(max_queue: usize, max_cc: usize, delay: Duration) -> Arc<Scheduler> {
        let backend: Arc<dyn Backend> = Arc::new(StubBackend {
            per_token_delay: delay,
        });
        Scheduler::spawn(
            backend,
            SchedulerConfig {
                max_queue_size: max_queue,
                max_concurrency: max_cc,
            },
        )
    }

    #[tokio::test]
    async fn queue_full_rejects_with_busy() {
        let s = spawn_stub(1, 1, Duration::from_millis(30)).await;
        let a = s.submit(params(5)).unwrap();
        // Wait until `a` is actually running (inflight == 1).
        wait_for(|| s.stats().inflight == 1).await;
        let b = s.submit(params(5)).unwrap(); // queued (queue=1/1)
        let e = s.submit(params(5)).err().expect("queue full must reject"); // queue full
        assert!(matches!(e, EdgeError::Busy(_)));
        assert_eq!(s.stats().queue_full_rejections, 1);
        drop((a, b));
        s.shutdown();
    }

    #[tokio::test]
    async fn cancel_removes_queued_entry_without_running_it() {
        let s = spawn_stub(4, 1, Duration::from_millis(30)).await;
        let a = s.submit(params(6)).unwrap();
        wait_for(|| s.stats().inflight == 1).await;
        let b = s.submit(params(6)).unwrap(); // queued
        assert_eq!(s.stats().queue_depth, 1);
        assert!(s.cancel(b.id), "queued request is removed");
        assert_eq!(s.stats().queue_depth, 0);
        // `b` must never produce events: its receiver gets Done(Err)?? No —
        // it was removed, so the channel simply closes with no events.
        drop(a.rx);
        wait_for(|| s.stats().inflight == 0).await;
        assert_eq!(s.stats().completed, 0, "a was dropped, not completed");
    }

    #[tokio::test]
    async fn dropping_receiver_cancels_running_generation() {
        let s = spawn_stub(4, 1, Duration::from_millis(20)).await;
        {
            let a = s.submit(params(50)).unwrap();
            wait_for(|| s.stats().inflight == 1).await;
            let _keep = a.alive; // hold the liveness receiver like a handler would
            drop(a.rx);
            // The blocking thread notices the dead channel on the next token
            // send and stops; the run is accounted as cancelled.
            wait_for(|| s.stats().cancelled == 1).await;
        }
        // Capacity is free again.
        let b = s.submit(params(2)).unwrap();
        let mut events = Vec::new();
        let mut rx = b.rx;
        while let Some(ev) = rx.recv().await {
            events.push(ev);
            if matches!(events.last(), Some(StreamEvent::Done(_))) {
                break;
            }
        }
        assert!(matches!(events.last(), Some(StreamEvent::Done(Ok(_)))));
        s.shutdown();
    }

    #[tokio::test]
    async fn done_event_carries_output_and_counters() {
        let s = spawn_stub(4, 2, Duration::from_millis(1)).await;
        let h = s.submit(params(3)).unwrap();
        let mut tokens = Vec::new();
        let mut rx = h.rx;
        loop {
            match rx.recv().await {
                Some(StreamEvent::Token(t)) => tokens.push(t),
                Some(StreamEvent::Done(Ok(out))) => {
                    assert_eq!(out.completion_tokens, 3);
                    assert_eq!(out.finish_reason, FinishReason::Length);
                    break;
                }
                other => panic!("unexpected event {other:?}"),
            }
        }
        assert_eq!(tokens, vec!["t0 ", "t1 ", "t2 "]);
        wait_for(|| s.stats().completed == 1).await;
        s.shutdown();
    }

    async fn wait_for(mut cond: impl FnMut() -> bool) {
        for _ in 0..500 {
            if cond() {
                return;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        panic!("condition not reached in 5s");
    }
}
