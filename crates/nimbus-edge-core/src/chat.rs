//! Chat templates.
//!
//! A template renders the `messages` array of /v1/chat/completions into the
//! exact prompt string the model was trained on. Templates are *named and
//! explicit*: model metadata selects one, and a model without a template
//! produces a clear error instead of an invented concatenation pretending to
//! be one.
//!
//! v1 ships the ChatML template (SmolLM2-compatible). GPT-2 / distilgpt2 are
//! base models without a chat template; importing them without
//! `--chat-template` keeps them usable through /v1/completions while
//! /v1/chat/completions fails with `no_chat_template`. Adding a template
//! means adding a variant here plus snapshot tests — nothing else.

use crate::error::EdgeError;

/// A named chat template. `as_meta_str` values are what `edge-meta.json`
/// stores (`chat_template` field) and what `models import --chat-template`
/// accepts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatTemplate {
    None,
    ChatMl,
}

impl ChatTemplate {
    pub fn parse(s: &str) -> Result<ChatTemplate, EdgeError> {
        match s {
            "none" => Ok(ChatTemplate::None),
            "chatml" => Ok(ChatTemplate::ChatMl),
            other => Err(EdgeError::invalid(format!(
                "unknown chat template {other:?}; supported: none, chatml"
            ))),
        }
    }

    pub fn as_meta_str(&self) -> &'static str {
        match self {
            ChatTemplate::None => "none",
            ChatTemplate::ChatMl => "chatml",
        }
    }

    /// Render `messages` into a prompt. `generation_prompt` appends whatever
    /// makes the model start an assistant reply.
    pub fn render(&self, messages: &[RenderedMessage]) -> Result<String, EdgeError> {
        match self {
            ChatTemplate::None => Err(EdgeError::NoChatTemplate(String::new())),
            ChatTemplate::ChatMl => Ok(render_chatml(messages, true)),
        }
    }

    /// Stop sequences that terminate an assistant reply for this template.
    /// Matched on decoded text, so they work regardless of how the vocab
    /// encodes the special markers.
    pub fn stop_sequences(&self) -> Vec<String> {
        match self {
            ChatTemplate::None => Vec::new(),
            ChatTemplate::ChatMl => vec!["<|im_end|>".to_string()],
        }
    }
}

/// One validated message: role is one of system/user/assistant, content is a
/// plain string (validated upstream of the template).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenderedMessage {
    pub role: String,
    pub content: String,
}

impl RenderedMessage {
    pub fn new(role: &str, content: impl Into<String>) -> RenderedMessage {
        RenderedMessage {
            role: role.to_string(),
            content: content.into(),
        }
    }
}

/// ChatML as used by SmolLM2:
/// `<|im_start|>{role}\n{content}<|im_end|>\n` per message, then
/// `<|im_start|>assistant\n` when a generation prompt is requested.
/// No system message is injected implicitly — the caller sends one if the
/// conversation needs it.
fn render_chatml(messages: &[RenderedMessage], generation_prompt: bool) -> String {
    let mut out = String::new();
    for m in messages {
        out.push_str("<|im_start|>");
        out.push_str(&m.role);
        out.push('\n');
        out.push_str(&m.content);
        out.push_str("<|im_end|>\n");
    }
    if generation_prompt {
        out.push_str("<|im_start|>assistant\n");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn conversation() -> Vec<RenderedMessage> {
        vec![
            RenderedMessage::new("system", "You are a helpful assistant."),
            RenderedMessage::new("user", "What is the capital of France?"),
            RenderedMessage::new("assistant", "The capital of France is Paris."),
            RenderedMessage::new("user", "And its population?"),
        ]
    }

    #[test]
    fn chatml_snapshot_full_conversation() {
        // Snapshot: exact rendering for a real multi-turn conversation with a
        // system message. Any change to the format must update this string.
        assert_eq!(
            render_chatml(&conversation(), true),
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n\
             <|im_start|>user\nWhat is the capital of France?<|im_end|>\n\
             <|im_start|>assistant\nThe capital of France is Paris.<|im_end|>\n\
             <|im_start|>user\nAnd its population?<|im_end|>\n\
             <|im_start|>assistant\n"
        );
    }

    #[test]
    fn chatml_snapshot_single_user_message() {
        assert_eq!(
            render_chatml(&[RenderedMessage::new("user", "hello")], true),
            "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn chatml_no_implicit_system_message() {
        // Two user messages render exactly as sent — the template must not
        // invent a system turn.
        let msgs = vec![
            RenderedMessage::new("user", "a"),
            RenderedMessage::new("user", "b"),
        ];
        assert_eq!(
            render_chatml(&msgs, false),
            "<|im_start|>user\na<|im_end|>\n<|im_start|>user\nb<|im_end|>\n"
        );
    }

    #[test]
    fn chatml_keeps_empty_content_verbatim() {
        let msgs = vec![RenderedMessage::new("user", "")];
        assert_eq!(
            render_chatml(&msgs, true),
            "<|im_start|>user\n<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn chatml_stop_sequence_is_im_end() {
        assert_eq!(ChatTemplate::ChatMl.stop_sequences(), vec!["<|im_end|>"]);
    }

    #[test]
    fn none_template_refuses_to_render() {
        let err = ChatTemplate::None.render(&conversation()).unwrap_err();
        assert!(matches!(err, EdgeError::NoChatTemplate(_)));
        assert!(ChatTemplate::None.stop_sequences().is_empty());
    }

    #[test]
    fn meta_roundtrip() {
        for t in [ChatTemplate::None, ChatTemplate::ChatMl] {
            assert_eq!(ChatTemplate::parse(t.as_meta_str()).unwrap(), t);
        }
        assert!(ChatTemplate::parse("gibberish").is_err());
    }
}
