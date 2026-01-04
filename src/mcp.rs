use crate::model::{Part, Message, Role, MediaType};
use async_trait::async_trait;
use rmcp::model::{
    AnnotateAble, Annotated, CallToolRequestParam, CallToolResult, GetPromptRequestParam, GetPromptResult, Prompt,
    RawContent, ReadResourceRequestParam, ReadResourceResult, Resource, Tool, ResourceContents, PromptMessageContent
};
use rmcp::service::{RoleClient, RunningService};
use rmcp::ClientHandler;
use serde_json::{Value, json};
use std::ops::Deref;
use thiserror::Error;
use uuid::Uuid;

#[derive(Debug, Error)]
pub enum MCPError {
    #[error("MCP error: {0}")]
    Mcp(String),
    #[error("Tool not found: {0}")]
    ToolNotFound(String),
    #[error("Prompt not found: {0}")]
    PromptNotFound(String),
    #[error("Resource not found: {0}")]
    ResourceNotFound(String),
    #[error("Server not found: {0}")]
    ServerNotFound(String),
    #[error("Server ID mismatch")]
    ServerIdMismatch,
}

#[derive(Debug, Clone)]
pub struct Served<T> {
    pub value: T,
    pub server_id: String,
}

impl<T> Served<T> {
    pub fn new(value: T, server_id: String) -> Self {
        Self { value, server_id }
    }
}

pub trait Servable {
    fn served(self, id: String) -> Served<Self>
    where
        Self: Sized,
    {
        Served::new(self, id)
    }
}

impl<T: AnnotateAble> Servable for Annotated<T> {}
impl Servable for Tool {}
impl Servable for Prompt {}

/// Trait for MCP servers that can be used by the Agent.
#[async_trait]
pub trait MCPServer: Send + Sync {
    /// List available tools.
    async fn list_tools(&self) -> Result<Vec<Served<Tool>>, MCPError>;

    /// Execute a tool.
    async fn call_tool(&self, name: String, args: Value) -> Result<Part, MCPError>;

    /// List available prompts.
    async fn list_prompts(&self) -> Result<Vec<Served<Prompt>>, MCPError>;

    /// Get a prompt.
    async fn get_prompt(
        &self,
        prompt: &Served<Prompt>,
        args: Option<serde_json::Map<String, Value>>,
    ) -> Result<GetPromptResult, MCPError>;

    /// List available resources.
    async fn list_resources(&self) -> Result<Vec<Served<Resource>>, MCPError>;

    /// Read a resource.
    async fn read_resource(
        &self,
        resource: &Served<Resource>,
    ) -> Result<ReadResourceResult, MCPError>;

    /// Get a prompt and convert it to messages.
    async fn prompt(&self, name: &str, args: Value) -> Result<Vec<Message>, MCPError>;
}

pub struct MCPServerImpl<S: ClientHandler> {
    inner: RunningService<RoleClient, S>,
    id: String,
}

impl<S: ClientHandler> MCPServerImpl<S> {
    pub fn new(inner: RunningService<RoleClient, S>) -> Self {
        Self {
            inner,
            id: Uuid::new_v4().to_string(),
        }
    }
}

#[async_trait]
impl<S> MCPServer for MCPServerImpl<S>
where
    S: ClientHandler + Send + Sync + 'static,
{
    async fn list_tools(&self) -> Result<Vec<Served<Tool>>, MCPError> {
        let result = self
            .inner
            .deref()
            .list_tools(None)
            .await
            .map_err(|e| MCPError::Mcp(e.to_string()))?;
        Ok(result
            .tools
            .into_iter()
            .map(|t| t.served(self.id.clone()))
            .collect())
    }

    async fn call_tool(&self, name: String, args: Value) -> Result<Part, MCPError> {
        let params = CallToolRequestParam {
            name: name.clone().into(),
            arguments: args.as_object().cloned(),
        };

        let result = self
            .inner
            .deref()
            .call_tool(params)
            .await
            .map_err(|e| MCPError::Mcp(e.to_string()))?;

        let mut structured = json!({});
        let mut parts = Vec::new();
        let mut parsed_text_content: Option<Value> = None;
        let mut raw_text_content: Vec<String> = Vec::new();

        for content in result.content {
            match content.raw {
                RawContent::Text(text_content) => {
                    if let Ok(parsed) = serde_json::from_str::<Value>(&text_content.text) {
                        parsed_text_content = Some(parsed);
                    } else {
                        raw_text_content.push(text_content.text);
                    }
                }
                RawContent::Image(image_content) => {
                    parts.push(Part::Media {
                        media_type: MediaType::Image,
                        data: image_content.data,
                        mime_type: image_content.mime_type,
                        uri: None,
                        finished: true,
                    });
                }
                RawContent::Resource(resource) => {
                    parts.push(resource_to_part(resource.resource));
                }
                _ => {}
            }
        }

        if let Some(s) = result.structured_content {
            structured = s;
        } else if let Some(parsed) = parsed_text_content {
            structured = parsed;
        } else if !raw_text_content.is_empty() {
            structured = json!({ "response": raw_text_content });
        }

        Ok(Part::FunctionResponse {
            id: None,
            name,
            response: structured,
            parts,
            finished: true,
        })
    }

    async fn list_prompts(&self) -> Result<Vec<Served<Prompt>>, MCPError> {
        let result = self
            .inner
            .deref()
            .list_prompts(None)
            .await
            .map_err(|e| MCPError::Mcp(e.to_string()))?;
        Ok(result
            .prompts
            .into_iter()
            .map(|p| p.served(self.id.clone()))
            .collect())
    }

    async fn get_prompt(
        &self,
        prompt: &Served<Prompt>,
        args: Option<serde_json::Map<String, Value>>,
    ) -> Result<GetPromptResult, MCPError> {
        if prompt.server_id != self.id {
            return Err(MCPError::ServerIdMismatch);
        }
        let params = GetPromptRequestParam {
            name: prompt.value.name.clone().into(),
            arguments: args,
        };
        self.inner
            .deref()
            .get_prompt(params)
            .await
            .map_err(|e| MCPError::Mcp(e.to_string()))
    }

    async fn list_resources(&self) -> Result<Vec<Served<Resource>>, MCPError> {
        let result = self
            .inner
            .deref()
            .list_resources(None)
            .await
            .map_err(|e| MCPError::Mcp(e.to_string()))?;
        Ok(result
            .resources
            .into_iter()
            .map(|r| r.served(self.id.clone()))
            .collect())
    }

    async fn read_resource(
        &self,
        resource: &Served<Resource>,
    ) -> Result<ReadResourceResult, MCPError> {
        if resource.server_id != self.id {
            return Err(MCPError::ServerIdMismatch);
        }
        let params = ReadResourceRequestParam {
            uri: resource.value.uri.clone().into(),
        };
        self.inner
            .deref()
            .read_resource(params)
            .await
            .map_err(|e| MCPError::Mcp(e.to_string()))
    }

    async fn prompt(&self, name: &str, args: Value) -> Result<Vec<Message>, MCPError> {
        let prompts = self.list_prompts().await?;
        let prompt = prompts
            .iter()
            .find(|p| p.value.name == name)
            .ok_or_else(|| MCPError::PromptNotFound(name.to_string()))?;

        let result = self.get_prompt(prompt, args.as_object().cloned()).await?;
        
        let mut messages = Vec::new();
        for msg in result.messages {
            let role = match msg.role {
                rmcp::model::PromptMessageRole::User => Role::User,
                rmcp::model::PromptMessageRole::Assistant => Role::Assistant,
            };

            let part = match msg.content {
                PromptMessageContent::Text { text } => Part::Text { content: text, finished: true },
                PromptMessageContent::Image { image } => Part::Media { 
                    media_type: MediaType::Image,
                    data: image.data.clone(), 
                    mime_type: image.mime_type.clone(), 
                    uri: None,
                    finished: true 
                },
                PromptMessageContent::Resource { resource } => {
                    resource_to_part(resource.resource.clone())
                }
                _ => continue,
            };

            messages.push(match role {
                Role::User => Message::User(vec![part]),
                Role::Assistant => Message::Assistant(vec![part]),
            });
        }

        Ok(messages)
    }
}

/// A helper to combine multiple MCP servers into one.
pub struct MultiMCPServer {
    servers: Vec<Box<dyn MCPServer>>,
}

impl MultiMCPServer {
    pub fn new() -> Self {
        Self {
            servers: Vec::new(),
        }
    }

    pub fn add_server<S: ClientHandler + Send + Sync + 'static>(
        mut self,
        server: RunningService<RoleClient, S>,
    ) -> Self {
        self.servers.push(Box::new(MCPServerImpl::new(server)));
        self
    }

    pub fn add_boxed_server(mut self, server: Box<dyn MCPServer>) -> Self {
        self.servers.push(server);
        self
    }
}

#[async_trait]
impl MCPServer for MultiMCPServer {
    async fn list_tools(&self) -> Result<Vec<Served<Tool>>, MCPError> {
        let mut all_tools = Vec::new();
        for server in &self.servers {
            let tools = server.list_tools().await?;
            all_tools.extend(tools);
        }
        Ok(all_tools)
    }

    async fn call_tool(&self, name: String, args: Value) -> Result<Part, MCPError> {
        for server in &self.servers {
            let tools = server.list_tools().await?;
            if tools.iter().any(|t| t.value.name == name) {
                return server.call_tool(name, args).await;
            }
        }
        Err(MCPError::ToolNotFound(name))
    }

    async fn list_prompts(&self) -> Result<Vec<Served<Prompt>>, MCPError> {
        let mut all_prompts = Vec::new();
        for server in &self.servers {
            let prompts = server.list_prompts().await?;
            all_prompts.extend(prompts);
        }
        Ok(all_prompts)
    }

    async fn get_prompt(
        &self,
        prompt: &Served<Prompt>,
        args: Option<serde_json::Map<String, Value>>,
    ) -> Result<GetPromptResult, MCPError> {
        for server in &self.servers {
            match server.get_prompt(prompt, args.clone()).await {
                Ok(res) => return Ok(res),
                Err(MCPError::ServerIdMismatch) => continue,
                Err(e) => return Err(e),
            }
        }
        Err(MCPError::ServerNotFound(prompt.server_id.clone()))
    }

    async fn list_resources(&self) -> Result<Vec<Served<Resource>>, MCPError> {
        let mut all_resources = Vec::new();
        for server in &self.servers {
            let resources = server.list_resources().await?;
            all_resources.extend(resources);
        }
        Ok(all_resources)
    }

    async fn read_resource(
        &self,
        resource: &Served<Resource>,
    ) -> Result<ReadResourceResult, MCPError> {
        for server in &self.servers {
            match server.read_resource(resource).await {
                Ok(res) => return Ok(res),
                Err(MCPError::ServerIdMismatch) => continue,
                Err(e) => return Err(e),
            }
        }
        Err(MCPError::ServerNotFound(resource.server_id.clone()))
    }

    async fn prompt(&self, name: &str, args: Value) -> Result<Vec<Message>, MCPError> {
        for server in &self.servers {
            let prompts = server.list_prompts().await?;
            if prompts.iter().any(|p| p.value.name == name) {
                return server.prompt(name, args).await;
            }
        }
        Err(MCPError::PromptNotFound(name.to_string()))
    }
}

#[async_trait]
pub trait AttachResources {
    async fn resources(self, server: &dyn MCPServer, resources: Vec<Served<Resource>>) -> Result<Self, MCPError>
    where Self: Sized;
}

#[async_trait]
impl AttachResources for Message {
    async fn resources(mut self, server: &dyn MCPServer, resources: Vec<Served<Resource>>) -> Result<Self, MCPError> {
        for resource in resources {
            let result = server.read_resource(&resource).await?;
            for content in result.contents {
                self.parts_mut().push(resource_to_part(content));
            }
        }
        Ok(self)
    }
}

#[async_trait]
impl AttachResources for Vec<Message> {
    async fn resources(mut self, server: &dyn MCPServer, resources: Vec<Served<Resource>>) -> Result<Self, MCPError> {
        if !self.is_empty() {
            let first = self.remove(0);
            let first = first.resources(server, resources).await?;
            self.insert(0, first);
        }
        Ok(self)
    }
}

fn resource_to_part(resource: ResourceContents) -> Part {
    match resource {
        ResourceContents::TextResourceContents { text, mime_type, uri, .. } => {
             Part::Media {
                media_type: MediaType::Text,
                data: text,
                mime_type: mime_type.unwrap_or_else(|| "text/plain".to_string()),
                uri: Some(uri),
                finished: true,
            }
        }
        ResourceContents::BlobResourceContents { blob, mime_type, uri, .. } => {
            let mime = mime_type.unwrap_or_else(|| "application/octet-stream".to_string());
            let media_type = if mime.starts_with("image/") {
                MediaType::Image
            } else if mime == "application/pdf" {
                MediaType::Document
            } else {
                MediaType::Binary
            };

            Part::Media {
                media_type,
                data: blob,
                mime_type: mime,
                uri: Some(uri),
                finished: true,
            }
        }
    }
}
