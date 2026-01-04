//! Google Gemini API client implementation.

use async_trait::async_trait;
use base64::prelude::*;
use futures::{Stream, StreamExt, stream};
use reqwest::header::{HeaderMap, HeaderValue, CONTENT_TYPE};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use serde_with::skip_serializing_none;
use std::pin::Pin;

use crate::client::{Client, ClientError, StreamingClient};
use crate::http::{add_extra_headers, build_http_client, RequestBuilderExt, ResponseExt};
use crate::model::{FinishReason, Message, Part, Response, Usage, MediaType};
use crate::options::{ModelOptions, TransportOptions};
use crate::sse::SSEResponseExt;
use rmcp::model::CallToolResult;

/// Gemini model options.
#[skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GeminiModel {
    pub top_k: Option<u32>,
    pub safety_settings: Option<Vec<GeminiSafetySetting>>,
    pub stop_sequences: Option<Vec<String>>,
    pub response_mime_type: Option<String>,
    pub thinking_budget: Option<u32>,
    pub thinking_level: Option<GeminiThinkingLevel>,
    pub include_thoughts: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum GeminiThinkingLevel {
    Low,
    High,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeminiSafetySetting {
    pub category: String,
    pub threshold: String,
}

/// Gemini client.
#[derive(Debug, Clone)]
pub struct GeminiClient {
    api_key: String,
    base_url: String,
    model_options: ModelOptions<GeminiModel>,
    transport_options: TransportOptions,
}

impl GeminiClient {
    pub fn new(
        api_key: String,
        base_url: String,
        model_options: ModelOptions<GeminiModel>,
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
        if let Ok(error_resp) = serde_json::from_str::<GeminiErrorResponse>(body) {
            ClientError::ProviderError(format!(
                "Gemini error ({}): {}",
                error_resp.error.code, error_resp.error.message
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
        let model = self
            .model_options
            .model
            .clone()
            .ok_or_else(|| ClientError::Config("Model must be specified".to_string()))?;

        let method = if stream { "streamGenerateContent?alt=sse&" } else { "generateContent?" };
        let url = format!("{}/models/{}:{}key={}", self.base_url, model, method, self.api_key);

        let request_body = GeminiRequest::new(messages, &self.model_options, tools)?;

        let http_client = build_http_client(&self.transport_options)?;

        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

        let mut req = http_client.post(&url).headers(headers);
        req = add_extra_headers(req, &self.transport_options);
        
        Ok(req.json_logged(&request_body))
    }
}

#[async_trait]
impl Client for GeminiClient {
    type ModelProvider = GeminiModel;

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

        let gemini_response: GeminiResponse = response.json_logged().await?;
        Ok(gemini_response.into())
    }

    fn model_options(&self) -> &ModelOptions<Self::ModelProvider> {
        &self.model_options
    }

    fn transport_options(&self) -> &TransportOptions {
        &self.transport_options
    }
}

#[async_trait]
impl StreamingClient for GeminiClient {
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

        Ok(Box::pin(GeminiStream::new(response)))
    }
}

// --- Streaming Implementation ---

struct GeminiStream;

impl GeminiStream {
    fn new(response: reqwest::Response) -> impl Stream<Item = Result<Response, ClientError>> + Send {
        let sse_stream = response.sse();
        
        Box::pin(async_stream::try_stream! {
            let mut stream = Box::pin(sse_stream);
            let mut current_response = Response {
                data: vec![Message::Assistant(vec![])],
                usage: Usage::default(),
                finish: FinishReason::Unfinished,
            };
            
            #[derive(PartialEq)]
            enum PartType { Text, Reasoning, FunctionCall }
            let mut last_part_type: Option<PartType> = None;

            while let Some(event_result) = stream.next().await {
                let event_str = event_result?;
                
                let chunk_result: GeminiResponse = serde_json::from_str(&event_str)
                    .map_err(|e| ClientError::ProviderError(format!("JSON parse error: {}", e)))?;
                
                if let Some(usage_meta) = chunk_result.usage_metadata {
                    current_response.usage.prompt_tokens = Some(usage_meta.prompt_token_count);
                    current_response.usage.completion_tokens = Some(usage_meta.candidates_token_count.unwrap_or(0) + usage_meta.thoughts_token_count.unwrap_or(0));
                }

                if let Some(candidates) = chunk_result.candidates {
                    if let Some(candidate) = candidates.first() {
                        if let Some(content) = &candidate.content {
                            let parts = current_response.data[0].parts_mut();
                            
                            for part in &content.parts {
                                match part {
                                    GeminiPart::Text { text, thought } => {
                                        let is_thought = thought.unwrap_or(false);
                                        let current_type = if is_thought { PartType::Reasoning } else { PartType::Text };
                                        
                                        if let Some(last_type) = &last_part_type {
                                            if *last_type != current_type {
                                                if let Some(last_part) = parts.last_mut() {
                                                    match last_part {
                                                        Part::Text { finished, .. } => *finished = true,
                                                        Part::Reasoning { finished, .. } => *finished = true,
                                                        Part::FunctionCall { finished, .. } => *finished = true,
                                                        _ => {},
                                                    }
                                                }
                                            }
                                        }
                                        last_part_type = Some(current_type);

                                        let should_append = if let Some(last_part) = parts.last_mut() {
                                            match (last_part, is_thought) {
                                                (Part::Text { finished: false, .. }, false) => true,
                                                (Part::Reasoning { finished: false, .. }, true) => true,
                                                _ => false,
                                            }
                                        } else {
                                            false
                                        };

                                        if should_append {
                                            if let Some(last_part) = parts.last_mut() {
                                                match last_part {
                                                    Part::Text { content: t, .. } => t.push_str(text),
                                                    Part::Reasoning { content: c, .. } => c.push_str(text),
                                                    _ => {},
                                                }
                                            }
                                        } else {
                                            if is_thought {
                                                parts.push(Part::Reasoning {
                                                    content: text.clone(),
                                                    summary: None,
                                                    signature: None,
                                                    finished: false,
                                                });
                                            } else {
                                                parts.push(Part::Text {
                                                    content: text.clone(),
                                                    finished: false,
                                                });
                                            }
                                        }
                                    },
                                    GeminiPart::FunctionCall { function_call, thought_signature } => {
                                        if let Some(last_type) = &last_part_type {
                                            if *last_type != PartType::FunctionCall {
                                                 if let Some(last_part) = parts.last_mut() {
                                                    match last_part {
                                                        Part::Text { finished, .. } => *finished = true,
                                                        Part::Reasoning { finished, .. } => *finished = true,
                                                        _ => {},
                                                    }
                                                }
                                            }
                                        }
                                        last_part_type = Some(PartType::FunctionCall);
                                        
                                        parts.push(Part::FunctionCall {
                                            id: None,
                                            name: function_call.name.clone(),
                                            arguments: function_call.args.clone(),
                                            signature: thought_signature.clone(),
                                            finished: false,
                                        });
                                    },
                                    _ => {}
                                }
                            }
                        }

                        if let Some(finish_reason) = &candidate.finish_reason {
                            for part in current_response.data[0].parts_mut() {
                                match part {
                                    Part::Text { finished, .. } => *finished = true,
                                    Part::Reasoning { finished, .. } => *finished = true,
                                    Part::FunctionCall { finished, .. } => *finished = true,
                                    Part::FunctionResponse { finished, .. } => *finished = true,
                                    Part::Media { finished, .. } => *finished = true,
                                }
                            }

                            current_response.finish = match finish_reason.as_str() {
                                "STOP" => FinishReason::Stop,
                                "MAX_TOKENS" => FinishReason::OutputTokens,
                                "SAFETY" => FinishReason::ContentFilter,
                                "RECITATION" => FinishReason::ContentFilter,
                                _ => FinishReason::Stop,
                            };
                        }
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
struct GeminiRequest {
    contents: Vec<GeminiContent>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<GeminiTool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system_instruction: Option<GeminiContent>,
    generation_config: GeminiGenerationConfig,
    safety_settings: Option<Vec<GeminiSafetySetting>>,
}

#[derive(Debug, Serialize, Deserialize)]
struct GeminiContent {
    role: String,
    parts: Vec<GeminiPart>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all_fields = "camelCase")]
#[serde(untagged)]
enum GeminiPart {
    Text { 
        text: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        thought: Option<bool>,
    },
    FunctionCall { 
        function_call: GeminiFunctionCall,
        #[serde(skip_serializing_if = "Option::is_none")]
        thought_signature: Option<String>,
    },
    FunctionResponse { function_response: GeminiFunctionResponse },
    InlineData { inline_data: GeminiInlineData },
}

#[derive(Debug, Serialize, Deserialize)]
struct GeminiFunctionCall {
    name: String,
    args: Value,
}

#[derive(Debug, Serialize, Deserialize)]
struct GeminiFunctionResponse {
    name: String,
    response: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    parts: Option<Vec<GeminiFunctionResponsePart>>,
}

#[derive(Debug, Serialize, Deserialize)]
struct GeminiFunctionResponsePart {
    #[serde(rename = "inlineData")]
    inline_data: GeminiFunctionResponseBlob,
}

#[derive(Debug, Serialize, Deserialize)]
struct GeminiFunctionResponseBlob {
    #[serde(rename = "mimeType")]
    mime_type: String,
    data: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct GeminiInlineData {
    mime_type: String,
    data: String,
}

#[derive(Debug, Serialize)]
struct GeminiTool {
    function_declarations: Vec<GeminiFunctionDeclaration>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct GeminiFunctionDeclaration {
    name: String,
    description: String,
    parameters_json_schema: Option<Value>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct GeminiGenerationConfig {
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<u32>,
    max_output_tokens: Option<u32>,
    stop_sequences: Option<Vec<String>>,
    response_mime_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking_config: Option<GeminiThinkingConfig>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct GeminiThinkingConfig {
    include_thoughts: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking_budget: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking_level: Option<GeminiThinkingLevel>,
}

impl GeminiRequest {
    fn new(
        messages_in: Vec<Message>,
        model_options: &ModelOptions<GeminiModel>,
        tool_defs: Vec<rmcp::model::Tool>,
    ) -> Result<Self, ClientError> {
        let mut contents = Vec::new();
        
        for msg in messages_in {
            let role = match msg {
                Message::User(_) => "user",
                Message::Assistant(_) => "model",
            };
            
            let mut parts = Vec::new();
            for part in msg.parts() {
                match part {
                    Part::Text { content: t, .. } => parts.push(GeminiPart::Text { text: t.clone(), thought: None }),
                    Part::Reasoning { content, .. } => parts.push(GeminiPart::Text { text: content.clone(), thought: Some(true) }),
                    Part::Media { data, mime_type, .. } => {
                        let anchor_text = part.anchor_media();
                        parts.push(GeminiPart::Text { text: anchor_text, thought: None });

                        parts.push(GeminiPart::InlineData {
                            inline_data: GeminiInlineData {
                                mime_type: mime_type.clone(),
                                data: data.clone(),
                            },
                        });
                    }
                    Part::FunctionCall { name, arguments, signature, .. } => {
                        parts.push(GeminiPart::FunctionCall {
                            function_call: GeminiFunctionCall {
                                name: name.clone(),
                                args: arguments.clone(),
                            },
                            thought_signature: signature.clone(),
                        });
                    }
                    Part::FunctionResponse { name, response, parts: inner_parts, .. } => {
                        let mut parts_vec = Vec::new();
                        
                        for part in inner_parts {
                            match part {
                                Part::Media { data, mime_type, .. } => {
                                    parts_vec.push(GeminiFunctionResponsePart {
                                        inline_data: GeminiFunctionResponseBlob {
                                            mime_type: mime_type.clone(),
                                            data: data.clone(),
                                        }
                                    });
                                }
                                _ => {}
                            }
                        }

                        let function_response_parts = if parts_vec.is_empty() { None } else { Some(parts_vec) };

                        parts.push(GeminiPart::FunctionResponse {
                            function_response: GeminiFunctionResponse {
                                name: name.clone(),
                                response: response.clone(),
                                parts: function_response_parts,
                            },
                        });
                    }
                }
            }
            
            if !parts.is_empty() {
                contents.push(GeminiContent {
                    role: role.to_string(),
                    parts,
                });
            }
        }

        let tools = if !tool_defs.is_empty() {
            vec![GeminiTool {
                function_declarations: tool_defs.into_iter().map(|t| GeminiFunctionDeclaration {
                    name: t.name.into_owned(),
                    description: t.description.map(|d| d.into_owned()).unwrap_or_default(),
                    parameters_json_schema: Some(Value::Object((*t.input_schema).clone())),
                }).collect(),
            }]
        } else {
            Vec::new()
        };

        let system_instruction = model_options.system.as_ref().map(|s| GeminiContent {
            role: "user".to_string(),
            parts: vec![GeminiPart::Text { text: s.clone(), thought: None }],
        });

        Ok(GeminiRequest {
            contents,
            tools,
            system_instruction,
            generation_config: GeminiGenerationConfig {
                temperature: model_options.temperature,
                top_p: model_options.top_p,
                top_k: model_options.provider.top_k,
                max_output_tokens: model_options.max_tokens,
                stop_sequences: model_options.provider.stop_sequences.clone(),
                response_mime_type: model_options.provider.response_mime_type.clone(),
                thinking_config: if model_options.reasoning.unwrap_or(false) || model_options.provider.include_thoughts.unwrap_or(false) {
                    Some(GeminiThinkingConfig {
                        include_thoughts: Some(true),
                        thinking_budget: model_options.provider.thinking_budget,
                        thinking_level: model_options.provider.thinking_level.clone(),
                    })
                } else {
                    None
                },
            },
            safety_settings: model_options.provider.safety_settings.clone(),
        })
    }
}

// --- Response Types ---

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiResponse {
    candidates: Option<Vec<GeminiCandidate>>,
    usage_metadata: Option<GeminiUsageMetadata>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiCandidate {
    content: Option<GeminiContent>,
    finish_reason: Option<String>,
    index: Option<u32>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiUsageMetadata {
    prompt_token_count: u32,
    candidates_token_count: Option<u32>,
    total_token_count: u32,
    thoughts_token_count: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct GeminiErrorResponse {
    error: GeminiError,
}

#[derive(Debug, Deserialize)]
struct GeminiError {
    code: u32,
    message: String,
    status: String,
}

impl From<GeminiResponse> for Response {
    fn from(resp: GeminiResponse) -> Self {
        let mut parts = Vec::new();
        let mut finish_reason = FinishReason::Unfinished;

        if let Some(mut candidates) = resp.candidates {
            if !candidates.is_empty() {
                let candidate = candidates.remove(0);
                if let Some(content) = candidate.content {
                    for part in content.parts {
                        match part {
                            GeminiPart::Text { text, thought } => {
                                if thought.unwrap_or(false) {
                                    parts.push(Part::Reasoning {
                                        content: text,
                                        summary: None,
                                        signature: None,
                                        finished: true,
                                    });
                                } else {
                                    parts.push(Part::Text { content: text, finished: true });
                                }
                            }
                            GeminiPart::FunctionCall { function_call, thought_signature } => {
                                parts.push(Part::FunctionCall {
                                    id: None,
                                    name: function_call.name,
                                    arguments: function_call.args,
                                    signature: thought_signature,
                                    finished: true,
                                });
                            }
                            GeminiPart::FunctionResponse { function_response } => {
                                let mut inner_parts = Vec::new();
                                if let Some(gemini_parts) = function_response.parts {
                                    for p in gemini_parts {
                                        inner_parts.push(Part::Media {
                                            media_type: MediaType::Binary, // Default to binary for response parts
                                            data: p.inline_data.data,
                                            mime_type: p.inline_data.mime_type,
                                            uri: None,
                                            finished: true,
                                        });
                                    }
                                }

                                parts.push(Part::FunctionResponse {
                                    id: None,
                                    name: function_response.name,
                                    response: function_response.response,
                                    parts: inner_parts,
                                    finished: true,
                                });
                            }
                            _ => {}
                        }
                    }
                }
                
                if let Some(reason) = candidate.finish_reason {
                    finish_reason = match reason.as_str() {
                        "STOP" => FinishReason::Stop,
                        "MAX_TOKENS" => FinishReason::OutputTokens,
                        "SAFETY" => FinishReason::ContentFilter,
                        "RECITATION" => FinishReason::ContentFilter,
                        _ => FinishReason::Stop,
                    };
                }
            }
        }

        let usage = resp.usage_metadata.map(|u| Usage {
            prompt_tokens: Some(u.prompt_token_count),
            completion_tokens: Some(u.candidates_token_count.unwrap_or(0) + u.thoughts_token_count.unwrap_or(0)),
        }).unwrap_or_default();

        Response {
            data: vec![Message::Assistant(parts)],
            usage,
            finish: finish_reason,
        }
    }
}
