//! Common data models for provider-agnostic LLM requests and responses.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

/// Role of the message sender.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum Role {
    User,
    Assistant,
}

/// A single message in a conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Message {
    Text {
        role: Role,
        content: String,
    },
    Reasoning {
        role: Role,
        content: String,
        summary: Option<String>,
        signature: Option<String>,
    },
    FunctionCall {
        name: String,
        arguments: Value,
        signature: Option<String>,
    },
    FunctionResponse {
        name: String,
        response: Value,
    },
}

impl Message {
    /// Get the role of the message.
    pub fn role(&self) -> &Role {
        match self {
            Message::Text { role, .. } => role,
            Message::Reasoning { role, .. } => role,
            Message::FunctionCall { .. } => &Role::Assistant,
            Message::FunctionResponse { .. } => &Role::User,
        }
    }

    /// Get the content of the message.
    pub fn content(&self) -> Option<String> {
        match self {
            Message::Text { content, .. } => Some(content.clone()),
            Message::Reasoning { content, .. } => Some(content.clone()),
            Message::FunctionCall { .. } => None,
            Message::FunctionResponse { .. } => None,
        }
    }
}

/// Provider-agnostic request structure.
/// Contains only model behavior parameters, not API configuration.
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, serde_json::Value>>,
}

/// Reason for finishing the response generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FinishReason {
    Stop,
    PromptTokens,
    OutputTokens,
    ToolCalls,
    ContentFilter,
    Error,
}

/// Token usage information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    /// Total prompt tokens used
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_tokens: Option<u32>,

    /// Total completion tokens used
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion_tokens: Option<u32>,
}

/// Provider-agnostic response structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Response {
    /// Generated messages (typically one assistant message, but can be multiple)
    pub data: Vec<Message>,

    /// Token usage information
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,

    /// Finish reason for the response generation
    pub finish: FinishReason,
}

/// Streaming response chunk - can be data, usage, or finish information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StreamChunk {
    /// Message data chunk
    Data(Message),
    
    /// Token usage information
    Usage(Usage),
    
    /// Finish reason
    Finish(FinishReason),
}
