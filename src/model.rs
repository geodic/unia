//! Common data models for provider-agnostic LLM requests and responses.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use serde_with::skip_serializing_none;
use std::collections::HashMap;
use rmcp::model::CallToolResult;

/// Role of the message sender.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum Role {
    User,
    Assistant,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MediaType {
    Image,
    Document,
    Text,
    Binary,
}

/// A part of a message content.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
pub enum Part {
    Text {
        content: String,
        #[serde(default)]
        finished: bool,
    },
    Reasoning {
        content: String,
        summary: Option<String>,
        signature: Option<String>,
        #[serde(default)]
        finished: bool,
    },
    FunctionCall {
        id: Option<String>,
        name: String,
        arguments: Value,
        signature: Option<String>,
        #[serde(default)]
        finished: bool,
    },
    FunctionResponse {
        id: Option<String>,
        name: String,
        response: Value,
        parts: Vec<Part>,
        #[serde(default)]
        finished: bool,
    },
    Media {
        media_type: MediaType,
        data: String,
        mime_type: String,
        #[serde(default)]
        uri: Option<String>,
        #[serde(default)]
        finished: bool,
    },
}

impl Part {
    pub fn anchor_media(&self) -> String {
        match self {
            Part::Media { mime_type, uri, .. } => {
                let uri_str = uri.as_deref().unwrap_or("unknown");
                format!("File ({}) at {}:", mime_type, uri_str)
            }
            _ => panic!("anchor_media called on non-Media part"),
        }
    }
}

/// A single message in a conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "role", content = "content")]
pub enum Message {
    #[serde(rename = "user")]
    User(Vec<Part>),
    #[serde(rename = "assistant")]
    Assistant(Vec<Part>),
}

impl Message {
    /// Get the role of the message.
    pub fn role(&self) -> Role {
        match self {
            Message::User(_) => Role::User,
            Message::Assistant(_) => Role::Assistant,
        }
    }

    /// Get the parts of the message.
    pub fn parts(&self) -> &Vec<Part> {
        match self {
            Message::User(parts) => parts,
            Message::Assistant(parts) => parts,
        }
    }

    /// Get the mutable parts of the message.
    pub fn parts_mut(&mut self) -> &mut Vec<Part> {
        match self {
            Message::User(parts) => parts,
            Message::Assistant(parts) => parts,
        }
    }

    /// Get the text content of the message (concatenated text parts).
    pub fn content(&self) -> Option<String> {
        let parts = self.parts();
        let text_parts: Vec<&str> = parts
            .iter()
            .filter_map(|p| match p {
                Part::Text { content: text, .. } => Some(text.as_str()),
                Part::Reasoning { content, .. } => Some(content.as_str()),
                _ => None,
            })
            .collect();

        if text_parts.is_empty() {
            None
        } else {
            Some(text_parts.join("\n"))
        }
    }
}

/// Provider-agnostic request structure.
/// Contains only model behavior parameters, not API configuration.
#[skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GeneralRequest {
    /// Model identifier (e.g., "gpt-4o", "claude-3-opus")
    pub model: Option<String>,

    /// Conversation history
    pub history: Vec<Message>,

    /// System instructions or prompt
    pub instructions: Option<String>,

    /// Maximum tokens to generate
    pub max_tokens: Option<u32>,

    /// Temperature for sampling (0.0 - 2.0)
    pub temperature: Option<f32>,

    /// Top-p sampling parameter
    pub top_p: Option<f32>,

    /// Arbitrary metadata for frontend/logging purposes
    pub metadata: Option<HashMap<String, serde_json::Value>>,
}

/// Reason for finishing the response generation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum FinishReason {
    Stop,
    PromptTokens,
    OutputTokens,
    ToolCalls,
    ContentFilter,
    Error,
    /// Default state when response is incomplete or streaming.
    /// If this is returned to the user, something went wrong.
    Unfinished,
}

/// Token usage information.
#[skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Usage {
    /// Total prompt tokens used
    pub prompt_tokens: Option<u32>,

    /// Total completion tokens used
    pub completion_tokens: Option<u32>,
}

impl std::ops::Add for Usage {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            prompt_tokens: self
                .prompt_tokens
                .map(|v| v + other.prompt_tokens.unwrap_or(0))
                .or(other.prompt_tokens),
            completion_tokens: self
                .completion_tokens
                .map(|v| v + other.completion_tokens.unwrap_or(0))
                .or(other.completion_tokens),
        }
    }
}

impl std::ops::AddAssign for Usage {
    fn add_assign(&mut self, other: Self) {
        *self = self.clone() + other;
    }
}

/// Provider-agnostic response structure.
#[skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Response {
    /// Generated messages (typically one assistant message, but can be multiple)
    pub data: Vec<Message>,

    /// Token usage information
    pub usage: Usage,

    /// Finish reason for the response generation
    pub finish: FinishReason,
}


