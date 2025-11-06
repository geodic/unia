//! Core client trait and error types.

use async_trait::async_trait;
use futures::Stream;
use thiserror::Error;

use crate::model::{Response, Message, StreamChunk};
use crate::options::{ModelOptions, TransportOptions};

/// Errors that can occur during client operations.
#[derive(Error, Debug)]
pub enum ClientError {
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("JSON parse error: {0}")]
    Parse(#[from] serde_json::Error),

    #[error("Provider error: {0}")]
    ProviderError(String),

    #[error("Stream cancelled")]
    StreamCancelled,

    #[error("Configuration error: {0}")]
    Config(String),
}

/// Main client trait for LLM providers.
///
/// Implement this trait to add support for a new LLM provider.
/// Each provider defines its own model and transport option types.
///
/// # Associated Types
/// - `ModelProvider`: Provider-specific model options (e.g., `OpenAiModel`)
/// - `TransportProvider`: Provider-specific transport options (e.g., `HttpTransport`)
///
/// # Required Methods
/// - `request`: Static method that sends a request with explicit options
/// - `new`: Constructor to create a client instance
/// - `model_options`: Accessor for the stored model options
/// - `transport_options`: Accessor for the stored transport options
///
/// # Provided Methods (with default implementations)
/// - `chat`: Uses default options
/// - `chat_with_options`: Overrides model options
///
/// # Example
/// ```rust,ignore
/// pub struct MyClient {
///     pub model_options: ModelOptions<MyModel>,
///     pub transport_options: TransportOptions<MyTransport>,
/// }
///
/// impl Client for MyClient {
///     type ModelProvider = MyModel;
///     type TransportProvider = MyTransport;
///     
///     async fn request(
///         messages: Vec<Message>,
///         model_options: &ModelOptions<Self::ModelProvider>,
///         transport_options: &TransportOptions<Self::TransportProvider>,
///     ) -> Result<GeneralResponse, ClientError> {
///         // Implementation
///     }
///     
///     fn new(
///         model_options: ModelOptions<Self::ModelProvider>,
///         transport_options: TransportOptions<Self::TransportProvider>,
///     ) -> Self {
///         Self { model_options, transport_options }
///     }
///     
///     fn model_options(&self) -> &ModelOptions<Self::ModelProvider> {
///         &self.model_options
///     }
///     
///     fn transport_options(&self) -> &TransportOptions<Self::TransportProvider> {
///         &self.transport_options
///     }
/// }
/// ```
#[async_trait]
pub trait Client: Send + Sync + Sized {
    /// Provider-specific model options type.
    type ModelProvider: Send + Sync;

    /// Provider-specific transport options type.
    type TransportProvider: Send + Sync;

    /// Core static request method that must be implemented by each provider.
    ///
    /// This is a static method that takes explicit options for full control.
    ///
    /// # Arguments
    /// - `messages`: Conversation messages
    /// - `model_options`: Model behavior options (temperature, max_tokens, etc.)
    /// - `transport_options`: Transport configuration (authentication, endpoints, etc.)
    ///
    /// # Returns
    /// Provider-agnostic response structure or error
    async fn request(
        messages: Vec<Message>,
        model_options: &ModelOptions<Self::ModelProvider>,
        transport_options: &TransportOptions<Self::TransportProvider>,
    ) -> Result<Response, ClientError>;

    /// Create a new client instance with the given options.
    ///
    /// # Arguments
    /// - `model_options`: Default model behavior options
    /// - `transport_options`: Default transport configuration
    ///
    /// # Returns
    /// A new client instance
    fn new(
        model_options: ModelOptions<Self::ModelProvider>,
        transport_options: TransportOptions<Self::TransportProvider>,
    ) -> Self;

    /// Get reference to the model options field.
    ///
    /// Typically just returns `&self.model_options`.
    fn model_options(&self) -> &ModelOptions<Self::ModelProvider>;

    /// Get reference to the transport options field.
    ///
    /// Typically just returns `&self.transport_options`.
    fn transport_options(&self) -> &TransportOptions<Self::TransportProvider>;

    /// Instance method that uses default options stored in the client.
    ///
    /// This is a convenience wrapper around `request` that uses the client's
    /// default model and transport options.
    ///
    /// # Arguments
    /// - `messages`: Conversation messages
    ///
    /// # Returns
    /// Provider-agnostic response structure or error
    async fn chat(&self, messages: Vec<Message>) -> Result<Response, ClientError> {
        Self::request(messages, self.model_options(), self.transport_options()).await
    }

    /// Instance method that overrides default model options.
    ///
    /// This is a convenience wrapper that uses the client's default transport options
    /// but allows you to override the model options for this specific request.
    ///
    /// # Arguments
    /// - `messages`: Conversation messages
    /// - `model_options`: Model options to use instead of defaults
    ///
    /// # Returns
    /// Provider-agnostic response structure or error
    async fn chat_with_options(
        &self,
        messages: Vec<Message>,
        model_options: &ModelOptions<Self::ModelProvider>,
    ) -> Result<Response, ClientError> {
        Self::request(messages, model_options, self.transport_options()).await
    }
}

/// Extension trait for streaming support.
///
/// Providers that support streaming should implement this trait in addition to `Client`.
/// This allows for a cleaner separation and avoids type inference issues.
///
/// # Required Methods
/// - `request_stream`: Static streaming method
///
/// # Provided Methods (with default implementations)
/// - `chat_stream`: Uses default options
/// - `chat_stream_with_options`: Overrides model options
///
/// # Example
/// ```rust,ignore
/// impl StreamingClient for MyClient {
///     async fn request_stream(
///         messages: Vec<Message>,
///         model_options: &ModelOptions<Self::ModelProvider>,
///         transport_options: &TransportOptions<Self::TransportProvider>,
///     ) -> Result<impl Stream<...>, ClientError> {
///         // Static streaming implementation
///     }
/// }
/// ```
#[async_trait]
pub trait StreamingClient: Client {
    /// Static streaming method.
    ///
    /// Returns a stream of chunks as the model generates the response.
    ///
    /// # Arguments
    /// - `messages`: Conversation messages
    /// - `model_options`: Model behavior options (temperature, max_tokens, etc.)
    /// - `transport_options`: Transport configuration (authentication, endpoints, etc.)
    ///
    /// # Returns
    /// A stream of `StreamChunk` items or an error
    async fn request_stream(
        messages: Vec<Message>,
        model_options: &ModelOptions<Self::ModelProvider>,
        transport_options: &TransportOptions<Self::TransportProvider>,
    ) -> Result<impl Stream<Item = Result<StreamChunk, ClientError>> + Send, ClientError>;

    /// Instance method for streaming that uses default options.
    ///
    /// This is a convenience wrapper around `request_stream` that uses the client's
    /// default model and transport options.
    ///
    /// # Arguments
    /// - `messages`: Conversation messages
    ///
    /// # Returns
    /// A stream of `StreamChunk` items or an error
    async fn chat_stream(
        &self,
        messages: Vec<Message>,
    ) -> Result<impl Stream<Item = Result<StreamChunk, ClientError>> + Send, ClientError> {
        Self::request_stream(
            messages,
            <Self as Client>::model_options(self),
            <Self as Client>::transport_options(self),
        )
        .await
    }

    /// Instance method for streaming that overrides default model options.
    ///
    /// This is a convenience wrapper that uses the client's default transport options
    /// but allows you to override the model options for this specific request.
    ///
    /// # Arguments
    /// - `messages`: Conversation messages
    /// - `model_options`: Model options to use instead of defaults
    ///
    /// # Returns
    /// A stream of `StreamChunk` items or an error
    async fn chat_stream_with_options(
        &self,
        messages: Vec<Message>,
        model_options: &ModelOptions<Self::ModelProvider>,
    ) -> Result<impl Stream<Item = Result<StreamChunk, ClientError>> + Send, ClientError> {
        Self::request_stream(
            messages,
            model_options,
            <Self as Client>::transport_options(self),
        )
        .await
    }
}
