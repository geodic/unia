//! OpenAI Responses API client implementation.
//!
//! This module implements the `Client` trait for OpenAI's Responses API using
//! the generic options architecture.
//! See: <https://platform.openai.com/docs/api-reference/responses>

use async_trait::async_trait;
use futures::Stream;
use itertools::Itertools;
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE};
use serde::{Deserialize, Serialize};

use crate::client::{Client, ClientError, StreamingClient};
use crate::http::{add_extra_headers, build_http_client};
use crate::model::{FinishReason, Response, Message, Role, Usage};
use crate::options::{HttpTransport, ModelOptions, OpenAiModel, TransportOptions};
use crate::sse::SSEResponseExt;

const DEFAULT_API_BASE: &str = "https://api.openai.com";
const DEFAULT_MODEL: &str = "gpt-5";

/// OpenAI client using HTTP transport.
pub struct OpenAiClient {
    model_options: ModelOptions<OpenAiModel>,
    transport_options: TransportOptions<HttpTransport>,
}

impl OpenAiClient {
    /// Create a new OpenAI client with default options.
    pub fn new(
        model_options: ModelOptions<OpenAiModel>,
        transport_options: TransportOptions<HttpTransport>,
    ) -> Self {
        Self {
            model_options,
            transport_options,
        }
    }

    /// Process streaming response from OpenAI.
    fn process_stream(
        response: reqwest::Response,
    ) -> impl Stream<Item = Result<crate::model::StreamChunk, ClientError>> + Send {
        use futures::StreamExt;
        use crate::model::StreamChunk;

        // Use the SSE response extension trait
        let sse_stream = response.sse().map(|result| {
            result.and_then(|line| {
                serde_json::from_str::<OpenAiStreamEvent>(&line).map_err(ClientError::Parse)
            })
        });

        // Map OpenAI-specific events to StreamChunk enum variants
        // Use flat_map to emit multiple chunks from the Done event (usage + finish)
        use futures::stream;
        
        sse_stream.flat_map(|result| {
            match result {
                Ok(event) => {
                    match event {
                        OpenAiStreamEvent::ResponseOutputTextDelta { delta } => {
                            stream::iter(vec![Ok(StreamChunk::Data(Message::Text {
                                role: Role::Assistant,
                                content: delta,
                            }))])
                        }
                        OpenAiStreamEvent::ResponseReasoningTextDelta { delta } => {
                            stream::iter(vec![Ok(StreamChunk::Data(Message::Reasoning {
                                role: Role::Assistant,
                                content: delta,
                                summary: None,
                                signature: None,
                            }))])
                        }
                        OpenAiStreamEvent::ResponseComplete { response } 
                        | OpenAiStreamEvent::ResponseIncomplete { response } => {
                            // Convert to Response using the existing From trait
                            let converted: Response = response.into();
                            let mut chunks = Vec::new();
                            
                            // Emit usage chunk if available
                            if let Some(usage) = converted.usage {
                                chunks.push(Ok(StreamChunk::Usage(usage)));
                            }

                            chunks.push(Ok(StreamChunk::Finish(converted.finish)));

                            stream::iter(chunks)
                        }
                        OpenAiStreamEvent::Other => {
                            stream::iter(vec![])
                        }
                    }
                }
                Err(e) => stream::iter(vec![Err(e)]),
            }
        })
    }

    /// Handle OpenAI error responses.
    fn handle_error_response(status: reqwest::StatusCode, body: &str) -> ClientError {
        if let Ok(error_resp) = serde_json::from_str::<OpenAiErrorResponse>(body) {
            ClientError::ProviderError(format!(
                "OpenAI error ({}): {}",
                error_resp.error.error_type, error_resp.error.message
            ))
        } else {
            ClientError::ProviderError(format!("HTTP {}: {}", status, body))
        }
    }
}

impl Default for OpenAiClient {
    fn default() -> Self {
        Self::new(
            ModelOptions {
                model: Some(DEFAULT_MODEL.to_string()),
                instructions: None,
                reasoning: None,
                temperature: None,
                top_p: None,
                max_tokens: None,
                provider: OpenAiModel {},
            },
            TransportOptions {
                timeout: None,
                provider: HttpTransport::default(),
            },
        )
    }
}

#[async_trait]
impl Client for OpenAiClient {
    type ModelProvider = OpenAiModel;
    type TransportProvider = HttpTransport;

    async fn request(
        messages: Vec<Message>,
        model_options: &ModelOptions<Self::ModelProvider>,
        transport_options: &TransportOptions<Self::TransportProvider>,
    ) -> Result<Response, ClientError> {
        // Validate API key is present
        let api_key = transport_options
            .provider
            .api_key
            .as_ref()
            .ok_or_else(|| ClientError::Config("API key is required".to_string()))?;

        let api_base = transport_options
            .provider
            .base_url
            .clone()
            .unwrap_or_else(|| DEFAULT_API_BASE.to_string());

        let url = format!("{}/v1/responses", api_base);
        let request_body = OpenAiRequest::from((messages, model_options));

        // Build HTTP client with transport options
        let http_client = build_http_client(transport_options)?;

        // Build request with extra headers if specified
        let mut req = http_client
            .post(&url)
            .header(AUTHORIZATION, format!("Bearer {}", api_key.expose_secret()))
            .header(CONTENT_TYPE, "application/json");

        req = add_extra_headers(req, &transport_options.provider.extra_headers);

        let response = req.json(&request_body).send().await?;
        let status = response.status();

        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(Self::handle_error_response(status, &body));
        }

        let openai_response: OpenAiResponse = response.json().await?;
        Ok(openai_response.into())
    }

    fn model_options(&self) -> &ModelOptions<Self::ModelProvider> {
        &self.model_options
    }

    fn transport_options(&self) -> &TransportOptions<Self::TransportProvider> {
        &self.transport_options
    }

    fn new(
        model_options: ModelOptions<Self::ModelProvider>,
        transport_options: TransportOptions<Self::TransportProvider>,
    ) -> Self {
        Self {
            model_options,
            transport_options,
        }
    }
}

#[async_trait]
impl StreamingClient for OpenAiClient {
    async fn request_stream(
        messages: Vec<Message>,
        model_options: &ModelOptions<Self::ModelProvider>,
        transport_options: &TransportOptions<Self::TransportProvider>,
    ) -> Result<impl Stream<Item = Result<crate::model::StreamChunk, ClientError>> + Send, ClientError> {
        // Validate API key is present
        let api_key = transport_options
            .provider
            .api_key
            .as_ref()
            .ok_or_else(|| ClientError::Config("API key is required".to_string()))?
            .expose_secret()
            .to_string();

        let api_base = transport_options
            .provider
            .base_url
            .clone()
            .unwrap_or_else(|| DEFAULT_API_BASE.to_string());

        let url = format!("{}/v1/responses", api_base);
        let mut request_body = OpenAiRequest::from((messages, model_options));
        request_body.stream = Some(true);

        // Build HTTP client with transport options
        let http_client = build_http_client(transport_options)?;

        // Build request with extra headers if specified
        let mut req = http_client
            .post(&url)
            .header(AUTHORIZATION, format!("Bearer {}", api_key))
            .header(CONTENT_TYPE, "application/json");

        req = add_extra_headers(req, &transport_options.provider.extra_headers);

        let response = req.json(&request_body).send().await?;
        let status = response.status();

        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(Self::handle_error_response(status, &body));
        }

        Ok(Self::process_stream(response))
    }
}

impl From<(Vec<Message>, &ModelOptions<OpenAiModel>)> for OpenAiRequest {
    fn from((messages, model_options): (Vec<Message>, &ModelOptions<OpenAiModel>)) -> Self {
        // Convert messages to OpenAI input format
        let input: Vec<OpenAiMessage> = messages.into_iter().map(|msg| msg.into()).collect();

        OpenAiRequest {
            model: model_options
                .model
                .clone()
                .unwrap_or_else(|| DEFAULT_MODEL.to_string()),
            input,
            instructions: model_options.instructions.clone(),
            max_output_tokens: model_options.max_tokens,
            temperature: model_options.temperature,
            top_p: model_options.top_p,
            stream: None,
        }
    }
}

impl From<Message> for OpenAiMessage {
    fn from(msg: Message) -> Self {
        match msg {
            Message::Text { role, content } => match role {
                Role::User => OpenAiMessage::Text {
                    content: vec![OpenAiContent {
                        text: content,
                        content_type: OpenAiContentType::InputText,
                    }],
                    role: OpenAiRole::User,
                },
                Role::Assistant => OpenAiMessage::Text {
                    content: vec![OpenAiContent {
                        text: content,
                        content_type: OpenAiContentType::OutputText,
                    }],
                    role: OpenAiRole::Assistant,
                },
            },
            Message::Reasoning {
                content, summary, ..
            } => OpenAiMessage::Reasoning {
                summary: vec![OpenAiContent {
                    text: summary.unwrap_or_default(),
                    content_type: OpenAiContentType::SummaryText,
                }],
                content: vec![OpenAiContent {
                    text: content,
                    content_type: OpenAiContentType::ReasoningText,
                }],
            },
            Message::FunctionCall { .. } | Message::FunctionResponse { .. } => {
                // OpenAI Responses API doesn't support function calls in the same way
                // Convert to text representation for now
                OpenAiMessage::Text {
                    content: vec![OpenAiContent {
                        text: String::new(),
                        content_type: OpenAiContentType::Text,
                    }],
                    role: OpenAiRole::Assistant,
                }
            }
        }
    }
}

impl From<OpenAiMessage> for Message {
    fn from(msg: OpenAiMessage) -> Self {
        match msg {
            OpenAiMessage::Text { content, role, .. } => {
                let message_role = match role {
                    OpenAiRole::User => Role::User,
                    OpenAiRole::Assistant => Role::Assistant,
                };
                Message::Text {
                    role: message_role,
                    // Could be replaced with map_or_default once stabilized
                    content: content.into_iter().map(|c| c.text).join("\n\n"), /* Must be pretty important seperation */
                }
            }
            OpenAiMessage::Reasoning {
                summary, content, ..
            } => Message::Reasoning {
                role: Role::Assistant,
                content: content.into_iter().map(|c| c.text).join("\n\n"),
                summary: Some(summary.into_iter().map(|c| c.text).join("\n\n"))
                    .filter(|s| !s.is_empty()),
                signature: None,
            },
        }
    }
}

impl From<OpenAiResponse> for Response {
    fn from(openai_resp: OpenAiResponse) -> Self {
        let messages = openai_resp
            .output
            .into_iter()
            .map(|message| message.into())
            .collect();

        Response {
            data: messages,
            usage: openai_resp.usage.map(|u| u.into()),
            finish: openai_resp.incomplete_details.map_or(FinishReason::Stop, |details| details.into()),
        }
    }
}

// --- OpenAI API Request/Response Types ---

#[derive(Debug, Clone, Serialize)]
struct OpenAiRequest {
    model: String,
    input: Vec<OpenAiMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    instructions: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
}

#[derive(Debug, Copy, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
enum OpenAiRole {
    User,
    Assistant,
}

#[derive(Debug, Copy, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum OpenAiContentType {
    InputText,
    OutputText,
    Text,
    SummaryText,
    ReasoningText,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OpenAiContent {
    text: String,
    #[serde(rename = "type")]
    content_type: OpenAiContentType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
enum OpenAiMessage {
    #[serde(rename = "message")]
    Text {
        content: Vec<OpenAiContent>,
        role: OpenAiRole,
    },
    Reasoning {
        summary: Vec<OpenAiContent>,
        content: Vec<OpenAiContent>,
    },
}

#[derive(Debug, Copy, Clone, Deserialize)]
#[serde(rename_all = "snake_case")]
enum IncompleteReason {
    MaxPromptTokens,
    MaxOutputTokens,
    ContentFilter,
    #[serde(other)]
    Other,
}

#[derive(Debug, Clone, Deserialize)]
struct IncompleteDetails {
    reason: IncompleteReason,
}

impl From<IncompleteDetails> for FinishReason {
    fn from(details: IncompleteDetails) -> Self {
        match details.reason {
            IncompleteReason::MaxPromptTokens => FinishReason::PromptTokens,
            IncompleteReason::MaxOutputTokens => FinishReason::OutputTokens,
            IncompleteReason::ContentFilter => FinishReason::ContentFilter,
            IncompleteReason::Other => FinishReason::Stop,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
struct OpenAiResponse {
    #[allow(dead_code)]
    id: String,
    #[allow(dead_code)]
    created_at: u64,
    #[allow(dead_code)]
    model: String,
    output: Vec<OpenAiMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<OpenAiUsage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    incomplete_details: Option<IncompleteDetails>,
}

#[derive(Debug, Copy, Clone, Deserialize)]
struct OpenAiUsage {
    input_tokens: u32,
    output_tokens: u32,
}

impl From<OpenAiUsage> for Usage {
    fn from(u: OpenAiUsage) -> Self {
        Usage {
            prompt_tokens: Some(u.input_tokens),
            completion_tokens: Some(u.output_tokens),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
struct OpenAiErrorResponse {
    error: OpenAiError,
}

#[derive(Debug, Clone, Deserialize)]
struct OpenAiError {
    #[serde(rename = "type")]
    error_type: String,
    message: String,
}

// --- OpenAI Streaming Response Types ---

/// Streaming event types from OpenAI
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type")]
enum OpenAiStreamEvent {
    #[serde(rename = "response.output_text.delta")]
    ResponseOutputTextDelta { delta: String },
    #[serde(rename = "response.reasoning_text.delta")]
    ResponseReasoningTextDelta { delta: String },
    #[serde(rename = "response.completed")]
    ResponseComplete {
        response: OpenAiResponse,
    },
    #[serde(rename = "response.incomplete")]
    ResponseIncomplete {
        response: OpenAiResponse,
    },
    #[serde(other)]
    Other,
}
