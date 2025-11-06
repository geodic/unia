//! # unai - Universal AI Client Library
//!
//! A small, pragmatic Rust library providing a provider-agnostic LLM client architecture
//! with a fully generic options system.
//!
//! ## Features
//! - Async-first, tokio compatible
//! - Provider-agnostic trait-based design
//! - Generic model and transport options
//! - Streaming support via Server-Sent Events
//! - Type-safe request/response models
//!
//! ## Architecture
//!
//! The library uses a two-tier API design:
//!
//! 1. **Static methods** for full control with explicit options
//! 2. **Instance methods** for convenience with stored default options
//!
//! ### Core Types
//!
//! - **`ModelOptions<T>`**: Model behavior parameters (temperature, max_tokens, etc.)
//! - **`TransportOptions<T>`**: Transport configuration (timeout, provider-specific settings)
//! - **Message**: Individual conversation messages with role and content
//!
//! ## Example
//! ```no_run
//! use unai::client::Client;
//! use unai::model::{Message, Role};
//! use unai::options::{HttpTransport, ModelOptions, OpenAiModel, SecretString, TransportOptions};
//! use unai::providers::OpenAiClient;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Set up options once
//!     let model_options = ModelOptions {
//!         model: Some("gpt-4o".to_string()),
//!         instructions: None,
//!         reasoning: None,
//!         temperature: Some(0.7),
//!         top_p: None,
//!         max_tokens: Some(100),
//!         provider: OpenAiModel {},
//!     };
//!     
//!     let transport_options = TransportOptions {
//!         timeout: None,
//!         provider: HttpTransport::new(SecretString::new("your-api-key".to_string())),
//!     };
//!     
//!     // Create client with default options
//!     let client = OpenAiClient::new(model_options, transport_options);
//!     
//!     // Use convenient instance method with just messages
//!     let messages = vec![
//!         Message::Text {
//!             role: Role::User,
//!             content: "Hello!".to_string(),
//!         }
//!     ];
//!     
//!     let response = client.chat(messages).await?;
//!     println!("{:?}", response);
//!     Ok(())
//! }
//! ```

pub mod client;
pub mod http;
pub mod model;
pub mod options;
pub mod providers;
pub mod sse;
pub mod stream;

// Re-exports for convenience
pub use client::{Client, ClientError, StreamingClient};
pub use model::{GeneralRequest, Response, Message};
pub use stream::StreamChunk;
