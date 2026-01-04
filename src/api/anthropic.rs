//! Anthropic API client implementation.

use async_trait::async_trait;
use base64::prelude::*;
use futures::{Stream, StreamExt, stream};
use reqwest::header::{HeaderMap, HeaderValue, CONTENT_TYPE};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use serde_with::skip_serializing_none;
use std::collections::HashMap;
use std::pin::Pin;

use crate::client::{Client, ClientError, StreamingClient};
use crate::http::{add_extra_headers, build_http_client, RequestBuilderExt, ResponseExt};
use crate::model::{FinishReason, Message, Part, Response, Usage, MediaType};
use crate::options::{ModelOptions, TransportOptions};
use crate::sse::SSEResponseExt;

const ANTHROPIC_VERSION: &str = "2023-06-01";

/// Anthropic model options.
#[skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AnthropicModel {
    pub top_k: Option<u32>,
    pub metadata: Option<AnthropicMetadata>,
    pub stop_sequences: Option<Vec<String>>,
    pub service_tier: Option<ServiceTier>,
    pub thinking_budget: Option<u32>,
    pub tool_choice: Option<AnthropicToolChoice>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicMetadata {
    pub user_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ServiceTier {
    Auto,
    StandardOnly,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnthropicToolChoice {
    Auto { disable_parallel_tool_use: Option<bool> },
    Any { disable_parallel_tool_use: Option<bool> },
    Tool { name: String, disable_parallel_tool_use: Option<bool> },
    None,
}

/// Anthropic client.
#[derive(Debug, Clone)]
pub struct AnthropicClient {
    api_key: String,
    base_url: String,
    model_options: ModelOptions<AnthropicModel>,
    transport_options: TransportOptions,
}

impl AnthropicClient {
    pub fn new(
        api_key: String,
        base_url: String,
        model_options: ModelOptions<AnthropicModel>,
        transport_options: TransportOptions,
    ) -> Self {
        Self {
            api_key,
            base_url,
            model_options,
            transport_options,
        }
    }

    fn handle_error_response(status: reqwest::StatusCode, body: &str) -> ClientError {
        if let Ok(error_resp) = serde_json::from_str::<AnthropicErrorResponse>(body) {
            ClientError::ProviderError(format!(
                "Anthropic error ({}): {}",
                error_resp.error.error_type, error_resp.error.message
            ))
        } else {
            ClientError::ProviderError(format!("HTTP {}: {}", status, body))
        }
    }

    fn build_request(
        &self,
        messages: Vec<Message>,
        tools: Vec<rmcp::model::Tool>,
        stream: bool,
    ) -> Result<reqwest::RequestBuilder, ClientError> {
        let url = format!("{}/messages", self.base_url);

        let model = self
            .model_options
            .model
            .clone()
            .ok_or_else(|| ClientError::Config("Model must be specified".to_string()))?;

        let request_body = AnthropicRequest::new(messages, &self.model_options, model, tools, stream);

        let http_client = build_http_client(&self.transport_options)?;

        let mut headers = HeaderMap::new();
        headers.insert(
            "x-api-key",
            HeaderValue::from_str(&self.api_key)
                .map_err(|_| ClientError::Config("Invalid API key".to_string()))?,
        );
        headers.insert(
            "anthropic-version",
            HeaderValue::from_static(ANTHROPIC_VERSION),
        );
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

        let mut req = http_client.post(&url).headers(headers);
        req = add_extra_headers(req, &self.transport_options);
        
        Ok(req.json_logged(&request_body))
    }
}

#[async_trait]
impl Client for AnthropicClient {
    type ModelProvider = AnthropicModel;

    async fn request(
        &self,
        messages: Vec<Message>,
        tools: Vec<rmcp::model::Tool>,
    ) -> Result<Response, ClientError> {
        let req = self.build_request(messages, tools, false)?;
        
        let response = req.send().await?;
        let status = response.status();

        if !status.is_success() {
            let body = response.text_logged().await.unwrap_or_default();
            return Err(Self::handle_error_response(status, &body));
        }

        let anthropic_response: AnthropicResponse = response.json_logged().await?;
        Ok(anthropic_response.into())
    }

    fn model_options(&self) -> &ModelOptions<Self::ModelProvider> {
        &self.model_options
    }

    fn transport_options(&self) -> &TransportOptions {
        &self.transport_options
    }
}

#[async_trait]
impl StreamingClient for AnthropicClient {
    async fn request_stream(
        &self,
        messages: Vec<Message>,
        tools: Vec<rmcp::model::Tool>,
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<Response, ClientError>> + Send>>,
        ClientError,
    > {
        let req = self.build_request(messages, tools, true)?;
        let response = req.send().await?;
        let status = response.status();

        if !status.is_success() {
            let body = response.text_logged().await.unwrap_or_default();
            return Err(Self::handle_error_response(status, &body));
        }

        Ok(Box::pin(AnthropicStream::new(response)))
    }
}

// --- Streaming Implementation ---

struct AnthropicStream;

impl AnthropicStream {
    fn new(response: reqwest::Response) -> impl Stream<Item = Result<Response, ClientError>> + Send {
        let sse_stream = response.sse();
        
        Box::pin(async_stream::try_stream! {
            let mut stream = Box::pin(sse_stream);
            let mut current_response = Response {
                data: vec![Message::Assistant(vec![])],
                usage: Usage::default(),
                finish: FinishReason::Unfinished,
            };
            
            let mut tool_buffers: HashMap<u32, (String, String, String)> = HashMap::new();

            while let Some(event_result) = stream.next().await {
                let event_str = event_result?;

                // Parse JSON event
                let chunk_result: AnthropicStreamEvent = serde_json::from_str(&event_str)
                    .map_err(|e| ClientError::ProviderError(format!("JSON parse error: {}", e)))?;

                match chunk_result {
                    AnthropicStreamEvent::MessageStart { message } => {
                        current_response.usage.prompt_tokens = Some(message.usage.input_tokens);
                        current_response.usage.completion_tokens = Some(message.usage.output_tokens);
                        yield current_response.clone();
                    },
                    AnthropicStreamEvent::ContentBlockStart { index, content_block } => {
                        let parts = current_response.data[0].parts_mut();
                        
                        match content_block {
                            AnthropicContentBlock::Text { text, .. } => {
                                parts.push(Part::Text { content: text, finished: false });
                            },
                            AnthropicContentBlock::ToolUse { id, name, .. } => {
                                tool_buffers.insert(index, (id.clone(), name.clone(), String::new()));
                                parts.push(Part::FunctionCall {
                                    id: Some(id),
                                    name,
                                    arguments: Value::Null,
                                    signature: None,
                                    finished: false,
                                });
                            },
                            AnthropicContentBlock::Thinking { thinking, signature } => {
                                parts.push(Part::Reasoning {
                                    content: thinking,
                                    summary: None,
                                    signature: Some(signature),
                                    finished: false,
                                });
                            },
                            _ => {},
                        }
                        yield current_response.clone();
                    },
                    AnthropicStreamEvent::ContentBlockDelta { index, delta } => {
                        let parts = current_response.data[0].parts_mut();
                        if let Some(part) = parts.get_mut(index as usize) {
                            match delta {
                                AnthropicDelta::TextDelta { text } => {
                                    if let Part::Text { content: current_text, .. } = part {
                                        current_text.push_str(&text);
                                    }
                                },
                                AnthropicDelta::InputJsonDelta { partial_json } => {
                                    if let Some(buffer) = tool_buffers.get_mut(&index) {
                                        buffer.2.push_str(&partial_json);
                                    }
                                },
                                AnthropicDelta::ThinkingDelta { thinking } => {
                                    if let Part::Reasoning { content, .. } = part {
                                        content.push_str(&thinking);
                                    }
                                },
                                AnthropicDelta::SignatureDelta { signature } => {
                                     if let Part::Reasoning { signature: sig, .. } = part {
                                        *sig = Some(signature);
                                    }
                                }
                            }
                        }
                        yield current_response.clone();
                    },
                    AnthropicStreamEvent::ContentBlockStop { index } => {
                        let parts = current_response.data[0].parts_mut();
                        if let Some(part) = parts.get_mut(index as usize) {
                            match part {
                                Part::Text { finished, .. } => *finished = true,
                                Part::Reasoning { finished, .. } => *finished = true,
                                Part::FunctionCall { finished, arguments, .. } => {
                                    *finished = true;
                                    if let Some((_, _, json_str)) = tool_buffers.remove(&index) {
                                        if let Ok(json_val) = serde_json::from_str(&json_str) {
                                            *arguments = json_val;
                                        }
                                    }
                                },
                                Part::FunctionResponse { finished, .. } => *finished = true,
                                Part::Media { finished, .. } => *finished = true,
                            }
                        }
                        yield current_response.clone();
                    },
                    AnthropicStreamEvent::MessageDelta { delta, usage } => {
                        if let Some(stop_reason) = delta.stop_reason {
                            current_response.finish = match stop_reason.as_str() {
                                "end_turn" => FinishReason::Stop,
                                "max_tokens" => FinishReason::OutputTokens,
                                "stop_sequence" => FinishReason::Stop,
                                "tool_use" => FinishReason::ToolCalls,
                                _ => FinishReason::Stop,
                            };
                        }
                        if let Some(usage_delta) = usage {
                            current_response.usage.completion_tokens = Some(usage_delta.output_tokens);
                        }
                        yield current_response.clone();
                    },
                    AnthropicStreamEvent::MessageStop => {
                        yield current_response.clone();
                    },
                    AnthropicStreamEvent::Ping => {},
                    AnthropicStreamEvent::Error { error } => {
                        Err(ClientError::ProviderError(format!("Stream error ({}): {}", error.error_type, error.message)))?;
                    }
                }
            }
        })
    }
}

// --- Request Types ---

#[skip_serializing_none]
#[derive(Debug, Serialize)]
struct AnthropicRequest {
    model: String,
    messages: Vec<AnthropicMessage>,
    max_tokens: u32,
    system: Option<Vec<AnthropicSystemBlock>>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<u32>,
    stream: Option<bool>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<AnthropicTool>,
    tool_choice: Option<AnthropicToolChoice>,
    metadata: Option<AnthropicMetadata>,
    stop_sequences: Option<Vec<String>>,
    service_tier: Option<ServiceTier>,
    thinking: Option<AnthropicThinkingConfig>,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AnthropicThinkingConfig {
    Enabled { budget_tokens: u32 },
    Disabled,
}

#[derive(Debug, Serialize)]
struct AnthropicTool {
    name: String,
    description: Option<String>,
    input_schema: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    cache_control: Option<AnthropicCacheControl>,
}

#[derive(Debug, Serialize)]
struct AnthropicMessage {
    role: String,
    content: Vec<AnthropicContentBlock>,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AnthropicSystemBlock {
    Text { 
        text: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<AnthropicCacheControl>,
    },
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AnthropicCacheControl {
    Ephemeral,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AnthropicToolResultBlock {
    Text { text: String },
    Image { source: AnthropicImageSource },
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
enum AnthropicToolResultContent {
    Text(String),
    Blocks(Vec<AnthropicToolResultBlock>),
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AnthropicContentBlock {
    Text { 
        text: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<AnthropicCacheControl>,
    },
    Image {
        source: AnthropicImageSource,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<AnthropicCacheControl>,
    },
    Document {
        source: AnthropicDocumentSource,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<AnthropicCacheControl>,
    },
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<AnthropicCacheControl>,
    },
    ToolResult {
        tool_use_id: String,
        content: AnthropicToolResultContent,
        #[serde(skip_serializing_if = "Option::is_none")]
        is_error: Option<bool>,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<AnthropicCacheControl>,
    },
    Thinking {
        thinking: String,
        signature: String,
    },
    RedactedThinking {
        data: String,
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct AnthropicImageSource {
    #[serde(rename = "type")]
    source_type: String,
    media_type: String,
    data: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct AnthropicDocumentSource {
    #[serde(rename = "type")]
    pub source_type: String,
    pub media_type: String,
    pub data: String,
}

impl AnthropicRequest {
    fn new(
        messages_in: Vec<Message>,
        model_options: &ModelOptions<AnthropicModel>,
        model: String,
        tool_defs: Vec<rmcp::model::Tool>,
        stream: bool,
    ) -> Self {
        let mut messages = Vec::new();
        
        for msg in messages_in {
            let role = match msg {
                Message::User(_) => "user",
                Message::Assistant(_) => "assistant",
            };

            let mut content_blocks = Vec::new();
            for part in msg.parts() {
                match part {
                    Part::Text { content: t, .. } => content_blocks.push(AnthropicContentBlock::Text { 
                        text: t.clone(),
                        cache_control: None,
                    }),
                    Part::Media { media_type, data, mime_type, .. } => {
                        content_blocks.push(AnthropicContentBlock::Text {
                            text: part.anchor_media(),
                            cache_control: None,
                        });

                        match media_type {
                            MediaType::Image => {
                                content_blocks.push(AnthropicContentBlock::Image {
                                    source: AnthropicImageSource {
                                        source_type: "base64".to_string(),
                                        media_type: mime_type.clone(),
                                        data: data.clone(),
                                    },
                                    cache_control: None,
                                });
                            },
                            MediaType::Document => {
                                content_blocks.push(AnthropicContentBlock::Document {
                                    source: AnthropicDocumentSource {
                                        source_type: "base64".to_string(),
                                        media_type: mime_type.clone(),
                                        data: data.clone(),
                                    },
                                    cache_control: None,
                                });
                            },
                            MediaType::Text | MediaType::Binary => {
                                let content = match BASE64_STANDARD.decode(data) {
                                    Ok(bytes) => String::from_utf8(bytes).unwrap_or(data.clone()),
                                    Err(_) => data.clone(),
                                };
                                content_blocks.push(AnthropicContentBlock::Text {
                                    text: content,
                                    cache_control: None,
                                });
                            }
                        }
                    }
                    Part::FunctionCall { id, name, arguments, .. } => {
                        if let Some(call_id) = id {
                            content_blocks.push(AnthropicContentBlock::ToolUse {
                                id: call_id.clone(),
                                name: name.clone(),
                                input: arguments.clone(),
                                cache_control: None,
                            });
                        }
                    }
                    Part::FunctionResponse { id, response, parts, .. } => {
                        if let Some(call_id) = id {
                            let mut blocks = Vec::new();
                            
                            if response.clone() != json!({}) {
                                blocks.push(AnthropicToolResultBlock::Text {
                                    text: serde_json::to_string(&response).unwrap_or_default(),
                                });
                            }

                            for part in parts {
                                match part {
                                    Part::Media { media_type, data, mime_type, .. } => {
                                        blocks.push(AnthropicToolResultBlock::Text {
                                            text: part.anchor_media(),
                                        });

                                        match media_type {
                                            MediaType::Image => {
                                                blocks.push(AnthropicToolResultBlock::Image {
                                                    source: AnthropicImageSource {
                                                        source_type: "base64".to_string(),
                                                        media_type: mime_type.clone(),
                                                        data: data.clone(),
                                                    },
                                                });
                                            },
                                            _ => {
                                                let content = match BASE64_STANDARD.decode(data) {
                                                    Ok(bytes) => String::from_utf8(bytes).unwrap_or(data.clone()),
                                                    Err(_) => data.clone(),
                                                };
                                                blocks.push(AnthropicToolResultBlock::Text {
                                                    text: content,
                                                });
                                            }
                                        }
                                    }
                                    _ => {}
                                }
                            }

                            content_blocks.push(AnthropicContentBlock::ToolResult {
                                tool_use_id: call_id.clone(),
                                content: AnthropicToolResultContent::Blocks(blocks),
                                is_error: None,
                                cache_control: None,
                            });
                        }
                    }
                    Part::Reasoning { content, signature, .. } => {
                        content_blocks.push(AnthropicContentBlock::Thinking {
                            thinking: content.clone(),
                            signature: signature.clone().unwrap_or_default(),
                        });
                    }
                }
            }
            
            if !content_blocks.is_empty() {
                messages.push(AnthropicMessage {
                    role: role.to_string(),
                    content: content_blocks,
                });
            }
        }

        let tools = tool_defs
            .into_iter()
            .map(|t| AnthropicTool {
                name: t.name.into_owned(),
                description: t.description.map(|d| d.into_owned()),
                input_schema: serde_json::Value::Object((*t.input_schema).clone()),
                cache_control: None,
            })
            .collect();

        let thinking = if model_options.reasoning.unwrap_or(false) {
            if let Some(budget) = model_options.provider.thinking_budget {
                Some(AnthropicThinkingConfig::Enabled { budget_tokens: budget })
            } else {
                Some(AnthropicThinkingConfig::Enabled { budget_tokens: 1024 })
            }
        } else {
            None
        };

        let system = model_options.system.as_ref().map(|s| vec![AnthropicSystemBlock::Text {
            text: s.clone(),
            cache_control: None,
        }]);

        AnthropicRequest {
            model,
            messages,
            max_tokens: model_options.max_tokens.unwrap_or(1024),
            system,
            temperature: model_options.temperature,
            top_p: model_options.top_p,
            top_k: model_options.provider.top_k,
            stream: if stream { Some(true) } else { None },
            tools,
            tool_choice: model_options.provider.tool_choice.clone(),
            metadata: model_options.provider.metadata.clone(),
            stop_sequences: model_options.provider.stop_sequences.clone(),
            service_tier: model_options.provider.service_tier.clone(),
            thinking,
        }
    }
}

// --- Response Types ---

#[derive(Debug, Deserialize)]
struct AnthropicResponse {
    id: String,
    #[serde(rename = "type")]
    msg_type: String,
    role: String,
    content: Vec<AnthropicContentBlock>,
    model: String,
    stop_reason: Option<String>,
    stop_sequence: Option<String>,
    usage: AnthropicUsage,
}

#[derive(Debug, Deserialize)]
struct AnthropicUsage {
    input_tokens: u32,
    output_tokens: u32,
    #[serde(default)]
    cache_creation_input_tokens: Option<u32>,
    #[serde(default)]
    cache_read_input_tokens: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct AnthropicErrorResponse {
    error: AnthropicError,
}

#[derive(Debug, Deserialize)]
struct AnthropicError {
    #[serde(rename = "type")]
    error_type: String,
    message: String,
}

impl From<AnthropicResponse> for Response {
    fn from(resp: AnthropicResponse) -> Self {
        let mut parts = Vec::new();
        
        for content in resp.content {
            match content {
                AnthropicContentBlock::Text { text, .. } => {
                    parts.push(Part::Text { content: text, finished: true });
                }
                AnthropicContentBlock::ToolUse { id, name, input, .. } => {
                    parts.push(Part::FunctionCall {
                        id: Some(id),
                        name,
                        arguments: input,
                        signature: None,
                        finished: true,
                    });
                }
                AnthropicContentBlock::Thinking { thinking, signature } => {
                    parts.push(Part::Reasoning {
                        content: thinking,
                        summary: None,
                        signature: Some(signature),
                        finished: true,
                    });
                }
                AnthropicContentBlock::RedactedThinking { .. } => {
                    // Skip redacted thinking
                }
                _ => {}
            }
        }

        let finish_reason = match resp.stop_reason.as_deref() {
            Some("end_turn") => FinishReason::Stop,
            Some("max_tokens") => FinishReason::OutputTokens,
            Some("stop_sequence") => FinishReason::Stop,
            Some("tool_use") => FinishReason::ToolCalls,
            _ => FinishReason::Stop,
        };

        Response {
            data: vec![Message::Assistant(parts)],
            usage: Usage {
                prompt_tokens: Some(resp.usage.input_tokens),
                completion_tokens: Some(resp.usage.output_tokens),
            },
            finish: finish_reason,
        }
    }
}

// --- SSE Event Types ---

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AnthropicStreamEvent {
    MessageStart { message: AnthropicResponse },
    ContentBlockStart { index: u32, content_block: AnthropicContentBlock },
    ContentBlockDelta { index: u32, delta: AnthropicDelta },
    ContentBlockStop { index: u32 },
    MessageDelta { delta: AnthropicMessageDelta, usage: Option<AnthropicUsage> },
    MessageStop,
    Ping,
    Error { error: AnthropicError },
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AnthropicDelta {
    TextDelta { text: String },
    InputJsonDelta { partial_json: String },
    ThinkingDelta { thinking: String },
    SignatureDelta { signature: String },
}

#[derive(Debug, Deserialize)]
struct AnthropicMessageDelta {
    stop_reason: Option<String>,
    stop_sequence: Option<String>,
}
