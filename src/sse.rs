//! Server-Sent Events (SSE) stream processing utilities.
//!
//! This module provides generic SSE parsing and stream processing
//! that can be shared across different LLM providers.
//!
//! SSE format:
//! ```text
//! data: {"key": "value"}
//!
//! data: {"another": "event"}
//!
//! data: [DONE]
//! ```

use futures::stream::{self, Stream, StreamExt};

use crate::client::ClientError;

/// Extension trait for `reqwest::Response` to enable SSE streaming.
///
/// This trait adds methods to easily convert HTTP responses into SSE event streams.
///
/// # Example
/// ```ignore
/// use unai::sse::SseResponseExt;
///
/// let response = client.get("https://api.example.com/stream").send().await?;
///
/// // Get raw SSE data lines
/// let mut stream = response.sse_lines();
/// while let Some(result) = stream.next().await {
///     let line = result?;
///     println!("SSE data: {}", line);
/// }
///
/// // Or automatically deserialize as JSON
/// let mut stream = response.sse_events::<MyEventType>();
/// while let Some(result) = stream.next().await {
///     let event = result?;
///     println!("Event: {:?}", event);
/// }
/// ```
pub trait SSEResponseExt {
    /// Convert the response into a stream of raw SSE data lines.
    ///
    /// Returns the content after `data: ` prefix for each SSE event.
    /// Stops when `[DONE]` marker is encountered or stream ends.
    fn sse(self) -> impl Stream<Item = Result<String, ClientError>> + Send;
}

impl SSEResponseExt for reqwest::Response {
    fn sse(self) -> impl Stream<Item = Result<String, ClientError>> + Send {
        let byte_stream = self.bytes_stream();

        stream::unfold(
            (Box::pin(byte_stream), String::new(), false),
            |(mut byte_stream, mut buffer, mut stream_ended)| async move {
                loop {
                    // If stream hasn't ended, try to read more data
                    if !stream_ended {
                        match byte_stream.next().await {
                            Some(Ok(chunk)) => {
                                // Append chunk to buffer
                                if let Ok(s) = std::str::from_utf8(&chunk) {
                                    buffer.push_str(s);
                                }
                            }
                            Some(Err(e)) => {
                                // HTTP error
                                return Some((Err(ClientError::from(e)), (byte_stream, buffer, stream_ended)));
                            }
                            None => {
                                // Byte stream ended - process any remaining complete lines in the buffer
                                stream_ended = true;
                            }
                        }
                    }

                    // Process complete lines from buffer
                    while let Some(pos) = buffer.find('\n') {
                        let line = buffer[..pos].trim().to_string();
                        buffer.drain(..=pos);

                        // Skip empty lines
                        if line.is_empty() {
                            continue;
                        }

                        // Parse SSE format: "data: {...}"
                        if let Some(data) = parse_sse_line(&line) {
                            // Check for stream end marker
                            if is_done_marker(data) {
                                return None;
                            }

                            // Return raw data line immediately
                            return Some((Ok(data.to_string()), (byte_stream, buffer, stream_ended)));
                        }
                    }

                    // If stream ended and no complete lines, try to process incomplete final line
                    if stream_ended {
                        if !buffer.is_empty() {
                            let line = buffer.trim().to_string();
                            buffer.clear();
                            if !line.is_empty() {
                                if let Some(data) = parse_sse_line(&line) {
                                    if !is_done_marker(data) {
                                        return Some((Ok(data.to_string()), (byte_stream, buffer, stream_ended)));
                                    }
                                }
                            }
                        }
                        
                        return None;
                    }

                    // No complete lines yet, continue reading
                }
            },
        )
    }
}

/// Parse an SSE line to extract the data portion.
///
/// SSE lines are in the format: `data: <content>`
///
/// # Example
/// ```
/// use unai::sse::parse_sse_line;
///
/// let line = "data: {\"key\": \"value\"}";
/// assert_eq!(parse_sse_line(line), Some("{\"key\": \"value\"}"));
///
/// let line = "invalid";
/// assert_eq!(parse_sse_line(line), None);
/// ```
pub fn parse_sse_line(line: &str) -> Option<&str> {
    line.strip_prefix("data: ").map(|s| s.trim())
}

/// Check if an SSE data line indicates the stream is done.
///
/// Common done marker: `[DONE]`
///
/// # Example
/// ```
/// use unai::sse::is_done_marker;
///
/// assert!(is_done_marker("[DONE]"));
/// assert!(!is_done_marker(""));
/// assert!(!is_done_marker("{\"data\": \"value\"}"));
/// ```
pub fn is_done_marker(data: &str) -> bool {
    data == "[DONE]"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_sse_line() {
        assert_eq!(parse_sse_line("data: hello"), Some("hello"));
        assert_eq!(
            parse_sse_line("data: {\"key\": \"value\"}"),
            Some("{\"key\": \"value\"}")
        );
        assert_eq!(parse_sse_line("data:   spaces  "), Some("spaces"));
        assert_eq!(parse_sse_line("invalid"), None);
        assert_eq!(parse_sse_line(""), None);
    }

    #[test]
    fn test_is_done_marker() {
        assert!(is_done_marker("[DONE]"));
        assert!(!is_done_marker(""));
        assert!(!is_done_marker("data"));
        assert!(!is_done_marker("{\"key\": \"value\"}"));
    }
}
