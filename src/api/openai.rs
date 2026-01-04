//! OpenAI Chat Completions API client implementation.

use async_trait::async_trait;
use futures::{Stream, StreamExt, stream};
use reqwest::header::{HeaderMap, HeaderValue, CONTENT_TYPE, AUTHORIZATION};
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

/// Trait for models compatible with OpenAI's Chat Completions API.
pub trait OpenAiCompatibleModel:
    Send + Sync + Default + Serialize + for<'de> Deserialize<'de> + Clone
{
}

/// Generic client for OpenAI-compatible Chat Completions APIs.
#[derive(Debug, Clone)]
pub struct OpenAiCompatibleClient<M> {
    api_key: String,
    base_url: String,
    model_options: ModelOptions<M>,
    transport_options: TransportOptions,
}

impl<M: OpenAiCompatibleModel> OpenAiCompatibleClient<M> {
    pub fn new(
        api_key: String,
        base_url: String,
        model_options: ModelOptions<M>,
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
        if let Ok(error_resp) = serde_json::from_str::<OpenAiErrorResponse>(body) {
            ClientError::ProviderError(format!(
                "OpenAI error ({}): {}",
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
        let url = format!("{}/chat/completions", self.base_url);

        let model = self
            .model_options
            .model
            .clone()
            .ok_or_else(|| ClientError::Config("Model must be specified".to_string()))?;

        let request_body = OpenAiRequest::new(messages, &self.model_options, model, tools, stream);

        let http_client = build_http_client(&self.transport_options)?;

        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {}", self.api_key))
                .map_err(|_| ClientError::Config("Invalid API key".to_string()))?,
        );

        let mut req = http_client.post(&url).headers(headers);
        req = add_extra_headers(req, &self.transport_options);
        
        Ok(req.json_logged(&request_body))
    }
}

#[async_trait]
impl<M: OpenAiCompatibleModel> Client for OpenAiCompatibleClient<M> {
    type ModelProvider = M;

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

        let openai_response: OpenAiResponse = response.json_logged().await?;
        Ok(openai_response.into())
    }

    fn model_options(&self) -> &ModelOptions<Self::ModelProvider> {
        &self.model_options
    }

    fn transport_options(&self) -> &TransportOptions {
        &self.transport_options
    }
}

#[async_trait]
impl<M: OpenAiCompatibleModel> StreamingClient for OpenAiCompatibleClient<M> {
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

        Ok(Box::pin(OpenAiStream::new(response)))
    }
}

// --- Streaming Implementation ---

struct OpenAiStream;

impl OpenAiStream {
    fn new(response: reqwest::Response) -> impl Stream<Item = Result<Response, ClientError>> + Send {
        let sse_stream = response.sse();
        
        Box::pin(async_stream::try_stream! {
            let mut stream = Box::pin(sse_stream);
            let mut current_response = Response {
                data: vec![Message::Assistant(vec![])],
                usage: Usage::default(),
                finish: FinishReason::Unfinished,
            };
            
            let mut tool_index_map: HashMap<u32, usize> = HashMap::new();
            let mut current_text_part_index: Option<usize> = None;

            while let Some(event_result) = stream.next().await {
                let event_str = event_result?;

                let chunk_result: OpenAiStreamChunk = serde_json::from_str(&event_str)
                    .map_err(|e| ClientError::ProviderError(format!("JSON parse error: {} | Input: {}", e, event_str)))?;

                if let Some(usage) = chunk_result.usage {
                    current_response.usage.prompt_tokens = Some(usage.prompt_tokens);
                    current_response.usage.completion_tokens = Some(usage.completion_tokens);
                }

                for choice in chunk_result.choices {
                    let parts = current_response.data[0].parts_mut();

                    if let Some(delta) = choice.delta {
                        if let Some(delta_content) = delta.content {
                            if let Some(idx) = current_text_part_index {
                                if let Some(Part::Text { content, .. }) = parts.get_mut(idx) {
                                    content.push_str(&delta_content);
                                }
                            } else {
                                parts.push(Part::Text { content: delta_content, finished: false });
                                current_text_part_index = Some(parts.len() - 1);
                            }
                        }

                        if let Some(tool_calls) = delta.tool_calls {
                            for tool_call in tool_calls {
                                let idx = *tool_index_map.entry(tool_call.index).or_insert_with(|| {
                                    parts.push(Part::FunctionCall {
                                        id: None,
                                        name: String::new(),
                                        arguments: Value::String(String::new()),
                                        signature: None,
                                        finished: false,
                                    });
                                    parts.len() - 1
                                });

                                if let Some(Part::FunctionCall { id: p_id, name: p_name, arguments: p_args, .. }) = parts.get_mut(idx) {
                                    if let Some(id) = tool_call.id {
                                        *p_id = Some(id);
                                    }
                                    if let Some(function) = tool_call.function {
                                        if let Some(name) = function.name {
                                            p_name.push_str(&name);
                                        }
                                        if let Some(args) = function.arguments {
                                            if let Value::String(arg_str) = p_args {
                                                arg_str.push_str(&args);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    if let Some(finish_reason) = choice.finish_reason {
                        for part in parts.iter_mut() {
                            match part {
                                Part::Text { finished, .. } => *finished = true,
                                Part::Reasoning { finished, .. } => *finished = true,
                                Part::FunctionCall { finished, arguments, .. } => {
                                    *finished = true;
                                    if let Value::String(json_str) = arguments {
                                        if let Ok(json_val) = serde_json::from_str(json_str) {
                                            *arguments = json_val;
                                        } else {
                                            *arguments = json!({});
                                        }
                                    }
                                },
                                Part::FunctionResponse { finished, .. } => *finished = true,
                                Part::Media { finished, .. } => *finished = true,
                            }
                        }

                        current_response.finish = match finish_reason.as_str() {
                            "stop" => FinishReason::Stop,
                            "length" => FinishReason::OutputTokens,
                            "tool_calls" => FinishReason::ToolCalls,
                            "content_filter" => FinishReason::ContentFilter,
                            _ => FinishReason::Stop,
                        };
                    }
                }
                
                yield current_response.clone();
            }
        })
    }
}

// --- Request Types ---

#[skip_serializing_none]
#[derive(Debug, Serialize)]
struct OpenAiRequest<M> {
    model: String,
    messages: Vec<OpenAiMessage>,
    max_tokens: Option<u32>, // OpenAI uses max_tokens or max_completion_tokens
    #[serde(rename = "max_completion_tokens")]
    max_completion_tokens: Option<u32>, // For o1/o3 models
    temperature: Option<f32>,
    top_p: Option<f32>,
    stream: Option<bool>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<OpenAiTool>,
    #[serde(flatten)]
    provider_options: M,
}

#[derive(Debug, Serialize)]
struct OpenAiMessage {
    role: String,
    content: OpenAiContent,
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tool_calls: Vec<OpenAiToolCall>,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
enum OpenAiContent {
    Text(String),
    Parts(Vec<OpenAiContentPart>),
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum OpenAiContentPart {
    Text { text: String },
    ImageUrl { image_url: OpenAiImageUrl },
    File { file: OpenAiFileContent },
}

#[derive(Debug, Serialize)]
struct OpenAiFileContent {
    #[serde(skip_serializing_if = "Option::is_none")]
    file_data: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    file_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    filename: Option<String>,
}

#[derive(Debug, Serialize)]
struct OpenAiImageUrl {
    url: String,
}

#[derive(Debug, Serialize)]
struct OpenAiTool {
    #[serde(rename = "type")]
    tool_type: String,
    function: OpenAiFunction,
}

#[derive(Debug, Serialize)]
struct OpenAiFunction {
    name: String,
    description: Option<String>,
    parameters: Value,
}

#[derive(Debug, Serialize, Deserialize)]
struct OpenAiToolCall {
    id: String,
    #[serde(rename = "type")]
    call_type: String,
    function: OpenAiFunctionCall,
}

#[derive(Debug, Serialize, Deserialize)]
struct OpenAiFunctionCall {
    name: String,
    arguments: String,
}

impl<M: OpenAiCompatibleModel> OpenAiRequest<M> {
    fn new(
        messages_in: Vec<Message>,
        model_options: &ModelOptions<M>,
        model: String,
        tool_defs: Vec<rmcp::model::Tool>,
        stream: bool,
    ) -> Self {
        let mut messages = Vec::new();
        
        // Handle system prompt as first message if present
        if let Some(system) = &model_options.system {
            messages.push(OpenAiMessage {
                role: "system".to_string(),
                content: OpenAiContent::Text(system.clone()),
                name: None,
                tool_call_id: None,
                tool_calls: Vec::new(),
            });
        }

        for msg in messages_in {
            let role = match msg {
                Message::User(_) => "user",
                Message::Assistant(_) => "assistant",
            };

            let mut content_parts = Vec::new();
            let mut tool_calls = Vec::new();
            let mut tool_call_id = None;
            let name = None;

            for part in msg.parts() {
                match part {
                    Part::Text { content: t, .. } => content_parts.push(OpenAiContentPart::Text { text: t.clone() }),
                    Part::Media { media_type, data, mime_type, uri, .. } => {
                        let anchor_text = part.anchor_media();
                        content_parts.push(OpenAiContentPart::Text { text: anchor_text });

                        match media_type {
                            MediaType::Image => {
                                content_parts.push(OpenAiContentPart::ImageUrl {
                                    image_url: OpenAiImageUrl {
                                        url: format!("data:{};base64,{}", mime_type, data),
                                    },
                                });
                            },
                            _ => {
                                content_parts.push(OpenAiContentPart::File {
                                    file: OpenAiFileContent {
                                        file_data: Some(data.clone()),
                                        file_id: None,
                                        filename: uri.clone(),
                                    }
                                });
                            }
                        }
                    }
                    Part::FunctionCall { id, name: fn_name, arguments, .. } => {
                        if let Some(call_id) = id {
                            tool_calls.push(OpenAiToolCall {
                                id: call_id.clone(),
                                call_type: "function".to_string(),
                                function: OpenAiFunctionCall {
                                    name: fn_name.clone(),
                                    arguments: arguments.to_string(),
                                },
                            });
                        }
                    }
                    Part::FunctionResponse { id, response, parts, .. } => {
                        if let Some(call_id) = id {
                            tool_call_id = Some(call_id.clone());
                            
                            let mut content_str = String::new();
                            
                            if response != &serde_json::json!({}) {
                                content_str.push_str(&response.to_string());
                            }

                            for part in parts {
                                match part {
                                    Part::Media { media_type, mime_type, .. } => {
                                        let anchor_text = part.anchor_media();
                                        content_str.push_str(&format!("\n{}", anchor_text));

                                        match media_type {
                                            MediaType::Image => content_str.push_str("\n[Image Content]"),
                                            _ => content_str.push_str(&format!("\n[File: {}]", mime_type)),
                                        }
                                    }
                                    _ => {}
                                }
                            }

                            content_parts.push(OpenAiContentPart::Text { text: content_str });
                        }
                    }
                    _ => {}
                }
            }

            // Override role for tool response
            let final_role = if tool_call_id.is_some() { "tool" } else { role };

            // If content is simple text, use Text variant, else Parts
            let content = if content_parts.len() == 1 {
                if let OpenAiContentPart::Text { text } = &content_parts[0] {
                    OpenAiContent::Text(text.clone())
                } else {
                    OpenAiContent::Parts(content_parts)
                }
            } else if !content_parts.is_empty() {
                OpenAiContent::Parts(content_parts)
            } else {
                OpenAiContent::Text(String::new()) // Empty content (e.g. pure tool call)
            };

            messages.push(OpenAiMessage {
                role: final_role.to_string(),
                content,
                name,
                tool_call_id,
                tool_calls,
            });
        }

        let tools = tool_defs
            .into_iter()
            .map(|t| OpenAiTool {
                tool_type: "function".to_string(),
                function: OpenAiFunction {
                    name: t.name.into_owned(),
                    description: t.description.map(|d| d.into_owned()),
                    parameters: Value::Object((*t.input_schema).clone()),
                },
            })
            .collect();

        // Check if model is o1/o3 to use max_completion_tokens
        let is_reasoning_model = model.starts_with("o1") || model.starts_with("o3");
        let (max_tokens, max_completion_tokens) = if is_reasoning_model {
            (None, model_options.max_tokens)
        } else {
            (model_options.max_tokens, None)
        };

        OpenAiRequest {
            model,
            messages,
            max_tokens,
            max_completion_tokens,
            temperature: model_options.temperature,
            top_p: model_options.top_p,
            stream: if stream { Some(true) } else { None },
            tools,
            provider_options: model_options.provider.clone(),
        }
    }
}

// --- Response Types ---

#[derive(Debug, Deserialize)]
struct OpenAiResponse {
    id: String,
    choices: Vec<OpenAiChoice>,
    usage: Option<OpenAiUsage>,
}

#[derive(Debug, Deserialize)]
struct OpenAiChoice {
    message: OpenAiResponseMessage,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAiResponseMessage {
    role: String,
    content: Option<String>,
    tool_calls: Option<Vec<OpenAiToolCall>>,
}

#[derive(Debug, Deserialize)]
struct OpenAiUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
}

#[derive(Debug, Deserialize)]
struct OpenAiErrorResponse {
    error: OpenAiError,
}

#[derive(Debug, Deserialize)]
struct OpenAiError {
    #[serde(rename = "type")]
    error_type: String,
    message: String,
}

impl From<OpenAiResponse> for Response {
    fn from(resp: OpenAiResponse) -> Self {
        let mut parts = Vec::new();
        let mut finish_reason = FinishReason::Stop;

        if let Some(choice) = resp.choices.first() {
            if let Some(content) = &choice.message.content {
                parts.push(Part::Text { content: content.clone(), finished: true });
            }
            if let Some(tool_calls) = &choice.message.tool_calls {
                for tool_call in tool_calls {
                    parts.push(Part::FunctionCall {
                        id: Some(tool_call.id.clone()),
                        name: tool_call.function.name.clone(),
                        arguments: serde_json::from_str(&tool_call.function.arguments).unwrap_or(Value::Null),
                        signature: None,
                        finished: true,
                    });
                }
            }

            if let Some(reason) = &choice.finish_reason {
                finish_reason = match reason.as_str() {
                    "stop" => FinishReason::Stop,
                    "length" => FinishReason::OutputTokens,
                    "tool_calls" => FinishReason::ToolCalls,
                    "content_filter" => FinishReason::ContentFilter,
                    _ => FinishReason::Stop,
                };
            }
        }

        let usage = resp.usage.map(|u| Usage {
            prompt_tokens: Some(u.prompt_tokens),
            completion_tokens: Some(u.completion_tokens),
        }).unwrap_or_default();

        Response {
            data: vec![Message::Assistant(parts)],
            usage,
            finish: finish_reason,
        }
    }
}

// --- Stream Types ---

#[derive(Debug, Deserialize)]
struct OpenAiStreamChunk {
    id: String,
    choices: Vec<OpenAiStreamChoice>,
    usage: Option<OpenAiUsage>,
}

#[derive(Debug, Deserialize)]
struct OpenAiStreamChoice {
    delta: Option<OpenAiDelta>,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAiDelta {
    content: Option<String>,
    tool_calls: Option<Vec<OpenAiStreamToolCall>>,
}

#[derive(Debug, Deserialize)]
struct OpenAiStreamToolCall {
    index: u32,
    id: Option<String>,
    function: Option<OpenAiStreamFunction>,
}

#[derive(Debug, Deserialize)]
struct OpenAiStreamFunction {
    name: Option<String>,
    arguments: Option<String>,
}
