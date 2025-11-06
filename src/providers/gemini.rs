//! Google Gemini API client implementation.
//!
//! This module implements the `Client` trait for Google's Gemini API using
//! the generic options architecture.
//! See: <https://ai.google.dev/api/rest>

use async_trait::async_trait;
use futures::Stream;
use nonempty::NonEmpty;
use reqwest::header::CONTENT_TYPE;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::client::{Client, ClientError, StreamingClient};
use crate::http::{add_extra_headers, build_http_client};
use crate::model::{FinishReason, Response, Message, Role, Usage};
use crate::options::{GeminiModel, HttpTransport, ModelOptions, TransportOptions};
use crate::sse::SSEResponseExt;

const DEFAULT_API_BASE: &str = "https://generativelanguage.googleapis.com";
const DEFAULT_MODEL: &str = "gemini-2.0-flash-exp";

/// Gemini client using HTTP transport.
pub struct GeminiClient {
    model_options: ModelOptions<GeminiModel>,
    transport_options: TransportOptions<HttpTransport>,
}

impl GeminiClient {
    /// Create a new Gemini client with default options.
    pub fn new(
        model_options: ModelOptions<GeminiModel>,
        transport_options: TransportOptions<HttpTransport>,
    ) -> Self {
        Self {
            model_options,
            transport_options,
        }
    }

    /// Process streaming response from Gemini.
    fn process_stream(
        response: reqwest::Response,
    ) -> impl Stream<Item = Result<crate::model::StreamChunk, ClientError>> + Send {
        use futures::StreamExt;
        use crate::model::{StreamChunk, Usage};

        // Use the SSE response extension trait
        let sse_stream = response.sse().map(|result| {
            result.and_then(|line| {
                serde_json::from_str::<GeminiResponse>(&line).map_err(ClientError::Parse)
            })
        });

        // Map Gemini-specific chunks to StreamChunk enum variants
        sse_stream.flat_map(|result| {
            use futures::stream;
            
            match result {
                Ok(gemini_resp) => {
                    let mut chunks = Vec::new();
                    
                    // Extract message data chunks
                    for candidate in gemini_resp.candidates.iter() {
                        for part in &candidate.content.parts {
                            chunks.push(Ok(StreamChunk::Data(part.clone().into())));
                        }
                    }
                    
                    // Add usage chunk if available
                    if let Some(usage_metadata) = gemini_resp.usage_metadata {
                        chunks.push(Ok(StreamChunk::Usage(Usage {
                            prompt_tokens: Some(usage_metadata.prompt_token_count),
                            completion_tokens: Some(
                                usage_metadata.candidates_token_count.unwrap_or_default()
                                    + usage_metadata.thoughts_token_count.unwrap_or_default()
                            ),
                        })));
                    }

                    // Only emit finish chunk if finish_reason is present (arrives at the end in streaming)
                    if let Some(finish_reason) = gemini_resp.candidates.last().finish_reason {
                        chunks.push(Ok(StreamChunk::Finish(finish_reason.into())));
                    }

                    stream::iter(chunks)
                }
                Err(e) => stream::iter(vec![Err(e)]),
            }
        })
    }

    /// Handle Gemini error responses.
    fn handle_error_response(status: reqwest::StatusCode, body: &str) -> ClientError {
        if let Ok(error_resp) = serde_json::from_str::<GeminiErrorResponse>(body) {
            ClientError::ProviderError(format!(
                "Gemini error ({}): {}",
                error_resp.error.code, error_resp.error.message
            ))
        } else {
            ClientError::ProviderError(format!("HTTP {}: {}", status, body))
        }
    }
}

impl From<(Vec<Message>, &ModelOptions<GeminiModel>)> for GeminiRequest {
    fn from((messages, model_options): (Vec<Message>, &ModelOptions<GeminiModel>)) -> Self {
        // Convert messages to Gemini content format
        let contents: Vec<GeminiContent> = messages.into_iter().map(|msg| msg.into()).collect();

        GeminiRequest {
            contents,
            generation_config: Some(GeminiGenerationConfig {
                temperature: model_options.temperature,
                top_p: model_options.top_p,
                max_output_tokens: model_options.max_tokens,
                thinking_config: Some(GeminiThinkingConfig {
                    include_thoughts: model_options.reasoning,
                    thinking_budget: None,
                }),
            }),
        }
    }
}

impl From<Role> for GeminiRole {
    fn from(role: Role) -> Self {
        match role {
            Role::User => GeminiRole::User,
            Role::Assistant => GeminiRole::Model,
        }
    }
}

impl From<Message> for GeminiContent {
    fn from(msg: Message) -> Self {
        GeminiContent {
            role: msg.role().clone().into(),
            parts: match msg {
                Message::Text { content, .. } => vec![GeminiPart::Text {
                    thought: None,
                    text: content,
                }],
                Message::Reasoning { content, .. } => vec![GeminiPart::Text {
                    thought: Some(true),
                    text: content,
                }],
                Message::FunctionCall {
                    name,
                    arguments,
                    signature,
                } => vec![GeminiPart::FunctionCall {
                    thought_signature: signature,
                    function_call: FunctionCall {
                        name,
                        args: arguments,
                    },
                }],
                Message::FunctionResponse { name, response } => {
                    vec![GeminiPart::FunctionResponse {
                        function_response: FunctionResponse { name, response },
                    }]
                }
            },
        }
    }
}

impl From<GeminiPart> for Message {
    fn from(part: GeminiPart) -> Self {
        match part {
            GeminiPart::Text { thought, text } => {
                if thought.unwrap_or_default() {
                    Message::Reasoning {
                        role: Role::Assistant,
                        content: text,
                        signature: None,
                        summary: None,
                    }
                } else {
                    Message::Text {
                        role: Role::Assistant,
                        content: text,
                    }
                }
            }
            GeminiPart::FunctionCall {
                thought_signature,
                function_call,
            } => Message::FunctionCall {
                name: function_call.name,
                arguments: function_call.args,
                signature: thought_signature,
            },
            GeminiPart::FunctionResponse { function_response } => Message::FunctionResponse {
                name: function_response.name,
                response: function_response.response,
            },
        }
    }
}

impl From<GeminiResponse> for Response {
    fn from(gemini_resp: GeminiResponse) -> Self {
        let finish_reason = gemini_resp
            .candidates
            .last().finish_reason
            .unwrap_or(GeminiFinishReason::Stop)
            .into();
        let parts = gemini_resp
            .candidates
            .into_iter()
            .flat_map(|candidate| candidate.content.parts.into_iter());

        Response {
            data: parts.map(|part| part.into()).collect(),
            usage: gemini_resp.usage_metadata.map(|u| u.into()),
            finish: finish_reason,
        }
    }
}

impl Default for GeminiClient {
    fn default() -> Self {
        Self::new(
            ModelOptions {
                model: Some(DEFAULT_MODEL.to_string()),
                instructions: None,
                reasoning: None,
                temperature: None,
                top_p: None,
                max_tokens: None,
                provider: GeminiModel {},
            },
            TransportOptions {
                timeout: None,
                provider: HttpTransport::default(),
            },
        )
    }
}

#[async_trait]
impl Client for GeminiClient {
    type ModelProvider = GeminiModel;
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

        // Determine model: use model_options or default
        let model = model_options
            .model
            .clone()
            .unwrap_or_else(|| DEFAULT_MODEL.to_string());

        let url = format!(
            "{}/v1beta/models/{}:generateContent?key={}",
            api_base,
            model,
            api_key.expose_secret()
        );

        let request_body = GeminiRequest::from((messages, model_options));

        // Build HTTP client with transport options
        let http_client = build_http_client(transport_options)?;

        // Build request with extra headers if specified
        let mut req = http_client
            .post(&url)
            .header(CONTENT_TYPE, "application/json");

        req = add_extra_headers(req, &transport_options.provider.extra_headers);

        let response = req.json(&request_body).send().await?;
        let status = response.status();

        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(Self::handle_error_response(status, &body));
        }

        let gemini_response: GeminiResponse = response.json().await?;
        Ok(gemini_response.into())
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
impl StreamingClient for GeminiClient {
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

        // Determine model: use model_options or default
        let model = model_options
            .model
            .clone()
            .unwrap_or_else(|| DEFAULT_MODEL.to_string());

        // Use alt=sse parameter for true streaming with Server-Sent Events
        let url = format!(
            "{}/v1beta/models/{}:streamGenerateContent?alt=sse&key={}",
            api_base, model, api_key
        );

        let request_body = GeminiRequest::from((messages, model_options));

        // Build HTTP client with transport options
        let http_client = build_http_client(transport_options)?;

        // Build request with extra headers if specified
        let mut req = http_client
            .post(&url)
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

// --- Gemini API Request/Response Types ---

#[derive(Debug, Clone, Serialize)]
struct GeminiRequest {
    contents: Vec<GeminiContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    generation_config: Option<GeminiGenerationConfig>,
}

#[derive(Debug, Copy, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
enum GeminiRole {
    User,
    Model,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GeminiContent {
    role: GeminiRole,
    parts: Vec<GeminiPart>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct FunctionCall {
    name: String,
    args: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct FunctionResponse {
    name: String,
    response: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged, rename_all = "camelCase")]
enum GeminiPart {
    Text {
        thought: Option<bool>,
        text: String,
    },
    FunctionCall {
        thought_signature: Option<String>,
        function_call: FunctionCall,
    },
    FunctionResponse {
        function_response: FunctionResponse,
    },
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct GeminiThinkingConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    include_thoughts: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking_budget: Option<u32>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct GeminiGenerationConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking_config: Option<GeminiThinkingConfig>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiResponse {
    candidates: NonEmpty<GeminiCandidate>,
    #[allow(dead_code)]
    model_version: Option<String>,
    usage_metadata: Option<GeminiUsageMetadata>,
}

#[derive(Debug, Copy, Clone, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
enum GeminiFinishReason {
    Stop,
    MaxTokens,
    Safety,
    Language,
    Blocklist,
    ProhibitedContent,
    Spii,
    ImageSafety,
    ImageProhibitedContent,
    ImageRecitation,
    MalformedFunctionCall,
    UnexpectedToolCall,
    TooManyToolCalls,
    #[serde(other)]
    Other,
}

impl From<GeminiFinishReason> for FinishReason {
    fn from(reason: GeminiFinishReason) -> Self {
        match reason {
            GeminiFinishReason::Stop => FinishReason::Stop,
            GeminiFinishReason::MaxTokens => FinishReason::OutputTokens,
            GeminiFinishReason::Safety
            | GeminiFinishReason::Language
            | GeminiFinishReason::Blocklist
            | GeminiFinishReason::ProhibitedContent
            | GeminiFinishReason::Spii
            | GeminiFinishReason::ImageSafety
            | GeminiFinishReason::ImageProhibitedContent
            | GeminiFinishReason::ImageRecitation => FinishReason::ContentFilter,
            GeminiFinishReason::MalformedFunctionCall
            | GeminiFinishReason::UnexpectedToolCall
            | GeminiFinishReason::TooManyToolCalls => FinishReason::ToolCalls,
            GeminiFinishReason::Other => FinishReason::Stop,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiCandidate {
    content: GeminiContent,
    finish_reason: Option<GeminiFinishReason>,
}

#[derive(Debug, Copy, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiUsageMetadata {
    prompt_token_count: u32,
    candidates_token_count: Option<u32>,
    thoughts_token_count: Option<u32>,
}

impl From<GeminiUsageMetadata> for Usage {
    fn from(u: GeminiUsageMetadata) -> Self {
        Usage {
            prompt_tokens: Some(u.prompt_token_count),
            completion_tokens: Some(
                u.candidates_token_count.unwrap_or_default()
                    + u.thoughts_token_count.unwrap_or_default()
            ),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
struct GeminiErrorResponse {
    error: GeminiError,
}

#[derive(Debug, Clone, Deserialize)]
struct GeminiError {
    code: u32,
    message: String,
}
