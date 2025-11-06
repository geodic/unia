//! Generic options structures for model and transport configuration.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// A secret string type for sensitive data like API keys.
/// Prevents accidental logging or display of secrets.
#[derive(Clone)]
pub struct SecretString(String);

impl SecretString {
    /// Create a new secret string.
    pub fn new(s: String) -> Self {
        Self(s)
    }

    /// Get the underlying secret value.
    pub fn expose_secret(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Debug for SecretString {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("SecretString([REDACTED])")
    }
}

impl From<String> for SecretString {
    fn from(s: String) -> Self {
        Self::new(s)
    }
}

impl From<&str> for SecretString {
    fn from(s: &str) -> Self {
        Self::new(s.to_string())
    }
}

/// Generic model options containing common model behavior parameters
/// and provider-specific model configuration.
///
/// # Type Parameters
/// - `T`: Provider-specific model options type
///
/// # Example
/// ```rust
/// use unai::options::{ModelOptions, OpenAiModel};
///
/// let options = ModelOptions {
///     model: Some("gpt-4o".to_string()),
///     instructions: None,
///     reasoning: None,
///     temperature: Some(0.7),
///     top_p: Some(0.9),
///     max_tokens: Some(100),
///     provider: OpenAiModel {},
/// };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ModelOptions<T> {
    /// Model identifier (e.g., "gpt-4o", "claude-3-opus")
    pub model: Option<String>,

    // System instructions passed to the model
    pub instructions: Option<String>,

    /// Enable reasoning/thinking mode (for models that support it)
    pub reasoning: Option<bool>,

    /// Temperature for sampling (0.0 - 2.0)
    pub temperature: Option<f32>,

    /// Top-p (nucleus) sampling parameter
    pub top_p: Option<f32>,

    /// Maximum tokens to generate
    pub max_tokens: Option<u32>,

    /// Provider-specific model options
    pub provider: T,
}

/// Generic transport options containing truly generic transport fields
/// and provider-specific transport configuration.
///
/// # Type Parameters
/// - `T`: Provider-specific transport options type
///
/// # Example
/// ```rust
/// use unai::options::{TransportOptions, HttpTransport, SecretString};
/// use std::time::Duration;
///
/// let options = TransportOptions {
///     timeout: Some(Duration::from_secs(30)),
///     provider: HttpTransport {
///         api_key: Some(SecretString::new("sk-...".to_string())),
///         base_url: Some("https://api.openai.com".to_string()),
///         proxy: None,
///         extra_headers: None,
///     },
/// };
/// ```
#[derive(Debug, Clone)]
pub struct TransportOptions<T> {
    /// Request timeout (applies to all transports)
    pub timeout: Option<Duration>,

    /// Provider-specific transport options
    pub provider: T,
}

/// HTTP-specific transport options.
/// Used as the provider field in `TransportOptions<HttpTransport>`.
#[derive(Debug, Clone, Default)]
pub struct HttpTransport {
    /// API key for authentication
    pub api_key: Option<SecretString>,

    /// Base URL for API endpoints
    pub base_url: Option<String>,

    /// HTTP proxy URL
    pub proxy: Option<String>,

    /// Additional HTTP headers to include in requests
    pub extra_headers: Option<HashMap<String, String>>,
}

impl HttpTransport {
    /// Create new HTTP transport options with an API key.
    pub fn new(api_key: impl Into<SecretString>) -> Self {
        Self {
            api_key: Some(api_key.into()),
            base_url: None,
            proxy: None,
            extra_headers: None,
        }
    }

    /// Set the base URL.
    pub fn with_base_url(mut self, base_url: String) -> Self {
        self.base_url = Some(base_url);
        self
    }

    /// Set the proxy URL.
    pub fn with_proxy(mut self, proxy: String) -> Self {
        self.proxy = Some(proxy);
        self
    }

    /// Set extra headers.
    pub fn with_extra_headers(mut self, headers: HashMap<String, String>) -> Self {
        self.extra_headers = Some(headers);
        self
    }

    /// Add a single extra header.
    pub fn with_header(mut self, key: String, value: String) -> Self {
        self.extra_headers
            .get_or_insert_with(HashMap::new)
            .insert(key, value);
        self
    }
}

/// OpenAI-specific model options.
/// Currently empty, but can be extended with OpenAI-specific parameters
/// like `frequency_penalty`, `presence_penalty`, `logit_bias`, etc.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OpenAiModel {
    // Future OpenAI-specific fields:
    // pub frequency_penalty: Option<f32>,
    // pub presence_penalty: Option<f32>,
    // pub logit_bias: Option<HashMap<String, f32>>,
    // pub seed: Option<u32>,
}

/// Gemini-specific model options.
/// Currently empty, but can be extended with Gemini-specific parameters
/// like `top_k`, `safety_settings`, etc.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GeminiModel {
    // Future Gemini-specific fields:
    // pub top_k: Option<u32>,
    // pub safety_settings: Option<Vec<SafetySetting>>,
    // pub stop_sequences: Option<Vec<String>>,
}

impl<T> ModelOptions<T> {
    /// Create new model options with provider-specific configuration.
    pub fn new(provider: T) -> Self {
        Self {
            model: None,
            instructions: None,
            reasoning: None,
            temperature: None,
            top_p: None,
            max_tokens: None,
            provider,
        }
    }

    /// Set the model identifier.
    pub fn with_model(mut self, model: String) -> Self {
        self.model = Some(model);
        self
    }

    /// Set the temperature.
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Set top-p sampling parameter.
    pub fn with_top_p(mut self, top_p: f32) -> Self {
        self.top_p = Some(top_p);
        self
    }

    /// Set maximum tokens to generate.
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }
}

impl<T> TransportOptions<T> {
    /// Create new transport options with provider-specific configuration.
    pub fn new(provider: T) -> Self {
        Self {
            timeout: None,
            provider,
        }
    }

    /// Set the timeout.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }
}
