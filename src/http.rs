//! HTTP client utilities for making requests to LLM APIs.
//!
//! This module provides reusable HTTP client construction and
//! request building logic that can be shared across providers.

use reqwest::{Client, RequestBuilder};
use std::collections::HashMap;

use crate::options::{HttpTransport, TransportOptions};

/// Build a configured HTTP client from transport options.
///
/// This applies common configuration like timeouts and proxies.
///
/// # Example
/// ```ignore
/// let client = build_http_client(&transport_options)?;
/// ```
pub fn build_http_client(
    transport_options: &TransportOptions<HttpTransport>,
) -> Result<Client, reqwest::Error> {
    let mut builder = Client::builder();

    if let Some(timeout) = transport_options.timeout {
        builder = builder.timeout(timeout);
    }

    if let Some(proxy_url) = &transport_options.provider.proxy {
        if let Ok(proxy) = reqwest::Proxy::all(proxy_url) {
            builder = builder.proxy(proxy);
        }
    }

    builder.build()
}

/// Add extra headers to a request if specified in transport options.
///
/// # Example
/// ```ignore
/// let mut req = client.post(url);
/// req = add_extra_headers(req, &transport_options.provider.extra_headers);
/// ```
pub fn add_extra_headers(
    mut request: RequestBuilder,
    extra_headers: &Option<HashMap<String, String>>,
) -> RequestBuilder {
    if let Some(headers) = extra_headers {
        for (key, value) in headers {
            request = request.header(key, value);
        }
    }
    request
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::options::SecretString;
    use std::time::Duration;

    #[test]
    fn test_build_http_client() {
        let transport_options = TransportOptions {
            timeout: Some(Duration::from_secs(30)),
            provider: HttpTransport {
                api_key: Some(SecretString::new("test".to_string())),
                base_url: None,
                proxy: None,
                extra_headers: None,
            },
        };

        let client = build_http_client(&transport_options);
        assert!(client.is_ok());
    }

    #[test]
    fn test_build_http_client_with_proxy() {
        let transport_options = TransportOptions {
            timeout: None,
            provider: HttpTransport {
                api_key: Some(SecretString::new("test".to_string())),
                base_url: None,
                proxy: Some("http://proxy.example.com:8080".to_string()),
                extra_headers: None,
            },
        };

        let client = build_http_client(&transport_options);
        assert!(client.is_ok());
    }
}
