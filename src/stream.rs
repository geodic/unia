//! Streaming support types and utilities.

// Re-export the StreamChunk enum from model.rs
pub use crate::model::StreamChunk;

// SSE parsing utilities have been moved to the `sse` module.
// Re-export them here for convenience.
pub use crate::sse::{is_done_marker, parse_sse_line};
