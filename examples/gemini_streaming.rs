//! Streaming Gemini example using the new generic options architecture.
//!
//! Run with:
//! ```bash
//! export GEMINI_API_KEY="your-api-key"
//! cargo run --example gemini_streaming
//! ```

use futures::StreamExt;
use unai::client::StreamingClient;
use unai::model::{Message, Role};
use unai::options::{GeminiModel, HttpTransport, ModelOptions, SecretString, TransportOptions};
use unai::providers::GeminiClient;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Get API key from environment
    let api_key =
        std::env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY environment variable must be set");

    // Create model options with higher temperature for creativity
    let model_options = ModelOptions {
        model: Some("gemini-2.5-flash".to_string()),
        instructions: None,
        reasoning: Some(true),
        temperature: Some(0.9),
        top_p: Some(0.95),
        max_tokens: Some(1024),
        provider: GeminiModel {},
    };

    // Create transport options with HTTP transport
    let transport_options = TransportOptions {
        timeout: Some(std::time::Duration::from_secs(60)),
        provider: HttpTransport::new(SecretString::new(api_key)),
    };

    // Create the client with default options
    let client = GeminiClient::new(model_options, transport_options);

    // Create messages
    let messages = vec![Message::Text {
        role: Role::User,
        content: "Write a haiku about Rust programming.".to_string(),
    }];

    println!("Streaming response from Gemini...\n");

    // Send streaming request using the instance method
    match client.chat_stream(messages).await {
        Ok(stream) => {
            // Pin the stream for polling
            futures::pin_mut!(stream);

            print!("Response: ");

            while let Some(chunk_result) = stream.next().await {
                match chunk_result {
                    Ok(chunk) => {
                        use unai::model::StreamChunk;
                        
                        match chunk {
                            StreamChunk::Data(message) => {
                                // Print message content as it arrives
                                if let Some(content) = message.content() {
                                    print!("{}", content);
                                }

                                // Flush stdout to show text immediately
                                use std::io::Write;
                                std::io::stdout().flush()?;
                            }
                            StreamChunk::Usage(usage) => {
                                println!("\n\n=== Usage Information ===");
                                if let Some(prompt_tokens) = usage.prompt_tokens {
                                    println!("Prompt tokens: {}", prompt_tokens);
                                }
                                if let Some(completion_tokens) = usage.completion_tokens {
                                    println!("Completion tokens: {}", completion_tokens);
                                }
                            }
                            StreamChunk::Finish(reason) => {
                                println!("\n\n=== Stream Complete ===");
                                println!("Finish reason: {:?}", reason);
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("\nError in stream: {}", e);
                        return Err(e.into());
                    }
                }
            }
        }
        Err(e) => {
            eprintln!("Error starting stream: {}", e);
            return Err(e.into());
        }
    }

    Ok(())
}
