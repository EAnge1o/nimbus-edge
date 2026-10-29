//! SSE shaping for `stream=true`.
//!
//! Wire format (documented in the README):
//! - the first data event carries the assistant role in its delta
//! - one data event per generated delta, `data: {...}\n\n`
//! - the final data event has an empty delta plus `finish_reason`
//! - then `data: [DONE]`
//! - an error after the stream started is emitted as
//!   `data: {"error": {...}}` followed by `[DONE]`, so any client that
//!   understands the success stream also sees the failure.

use std::pin::Pin;
use std::task::{Context, Poll};

use axum::response::sse::{Event, KeepAlive};
use futures_util::{Stream, StreamExt as _};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use nimbus_edge_core::openai::ErrorBody;
use nimbus_edge_core::scheduler::StreamEvent;

/// The scheduler's event channel plus the liveness receiver. Holding the
/// oneshot receiver inside the SSE body means a client disconnect (which
/// drops the body) drops *both* receivers at once — the exact signal the
/// scheduler and the blocking generation thread use to stop.
pub struct CancelAwareStream {
    inner: ReceiverStream<StreamEvent>,
    _alive: tokio::sync::oneshot::Receiver<()>,
}

impl CancelAwareStream {
    pub fn new(
        rx: mpsc::Receiver<StreamEvent>,
        alive: tokio::sync::oneshot::Receiver<()>,
    ) -> CancelAwareStream {
        CancelAwareStream {
            inner: ReceiverStream::new(rx),
            _alive: alive,
        }
    }
}

impl Stream for CancelAwareStream {
    type Item = StreamEvent;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Pin::new(&mut self.inner).poll_next(cx)
    }
}

/// Shapes raw scheduler events into `chat.completion.chunk` events, carrying
/// the `role` field in the first delta the way OpenAI clients expect.
pub struct ShapedChatStream {
    inner: CancelAwareStream,
    id: String,
    model: String,
    created: u64,
    role_sent: bool,
}

impl ShapedChatStream {
    pub fn new(inner: CancelAwareStream, id: String, model: String, created: u64) -> Self {
        ShapedChatStream {
            inner,
            id,
            model,
            created,
            role_sent: false,
        }
    }
}

impl Stream for ShapedChatStream {
    type Item = Result<Event, axum::Error>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match Pin::new(&mut self.inner).poll_next(cx) {
            Poll::Ready(Some(StreamEvent::Token(delta))) => {
                let mut obj = serde_json::Map::new();
                if !self.role_sent {
                    obj.insert("role".into(), serde_json::json!("assistant"));
                    self.role_sent = true;
                }
                obj.insert("content".into(), serde_json::json!(delta));
                Poll::Ready(Some(Ok(chat_chunk(
                    &self.id,
                    self.created,
                    &self.model,
                    obj.into(),
                    None,
                ))))
            }
            Poll::Ready(Some(StreamEvent::Done(Ok(out)))) => {
                let mut obj = serde_json::Map::new();
                if !self.role_sent {
                    obj.insert("role".into(), serde_json::json!("assistant"));
                }
                let finish = out.finish_reason.as_str();
                Poll::Ready(Some(Ok(chat_chunk(
                    &self.id,
                    self.created,
                    &self.model,
                    obj.into(),
                    Some(finish),
                ))))
            }
            Poll::Ready(Some(StreamEvent::Done(Err(e)))) => Poll::Ready(Some(Ok(error_event(&e)))),
            Poll::Ready(None) => Poll::Ready(None),
            Poll::Pending => Poll::Pending,
        }
    }
}

/// Append the closing `data: [DONE]` after a shaped stream.
pub fn with_done<S>(stream: S) -> Pin<Box<dyn Stream<Item = Result<Event, axum::Error>> + Send>>
where
    S: Stream<Item = Result<Event, axum::Error>> + Send + 'static,
{
    let tail = futures_util::stream::once(async { Ok(done_event()) });
    Box::pin(stream.chain(tail))
}

/// One assistant chunk in the `chat.completion.chunk` shape.
pub fn chat_chunk(
    id: &str,
    created: u64,
    model: &str,
    delta: serde_json::Value,
    finish_reason: Option<&str>,
) -> Event {
    let body = serde_json::json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": finish_reason,
        }],
    });
    Event::default().data(body.to_string())
}

pub fn error_event(err: &nimbus_edge_core::error::EdgeError) -> Event {
    let body: ErrorBody = ErrorBody::from_error(err);
    Event::default().data(serde_json::to_string(&body).expect("error body serializes"))
}

pub fn done_event() -> Event {
    Event::default().data("[DONE]")
}

pub fn keep_alive() -> KeepAlive {
    KeepAlive::new()
        .interval(std::time::Duration::from_secs(15))
        .text("keep-alive")
}
