//! Simple OpenAI example using the new generic options architecture.
//!
//! Run with:
//! ```bash
//! export OPENAI_API_KEY="your-api-key"
//! cargo run --example openai_simple
//! ```

use unai::client::Client;
use unai::model::{Message, Role};
use unai::options::{HttpTransport, ModelOptions, OpenAiModel, SecretString, TransportOptions};
use unai::providers::OpenAiClient;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Get API key from environment
    let api_key =
        std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY environment variable must be set");

    // Create model options (these will be the defaults for the client)
    let model_options = ModelOptions {
        model: Some("gpt-5".to_string()),
        instructions: None,
        reasoning: Some(true),
        temperature: Some(0.7),
        top_p: None,
        max_tokens: Some(100),
        provider: OpenAiModel {},
    };

    // Create transport options with HTTP transport
    let transport_options = TransportOptions {
        timeout: None,
        provider: HttpTransport::new(SecretString::new(api_key)),
    };

    // Create the client with default options
    let client = OpenAiClient::new(model_options.clone(), transport_options.clone());

    // Create messages
    let messages = vec![Message::Text {
        role: Role::User,
        content: "What is the capital of France? Answer in one word.".to_string(),
    }];

    println!("Sending request to OpenAI...");

    // Send request using the convenient instance method
    match client.chat(messages).await {
        Ok(response) => {
            println!("\n=== Response ===");

            if let Some(usage) = &response.usage {
                if let Some(prompt_tokens) = usage.prompt_tokens {
                    println!("Prompt tokens: {}", prompt_tokens);
                }
                if let Some(completion_tokens) = usage.completion_tokens {
                    println!("Completion tokens: {}", completion_tokens);
                }
            }

            println!("Finish reason: {:?}", response.finish);

            println!("\n=== Messages ===");
            for (i, message) in response.data.iter().enumerate() {
                println!("Message {}: {:?}", i + 1, message);
            }
        }
        Err(e) => {
            eprintln!("Error: {}", e);
            return Err(e.into());
        }
    }

    // Example with conversation history
    println!("\n\n=== Multi-turn conversation ===");

    let conversation_messages = vec![
        Message::Text {
            role: Role::User,
            content: "My name is Alice.".to_string(),
        },
        Message::Text {
            role: Role::Assistant,
            content: "Hello Alice! Nice to meet you.".to_string(),
        },
        Message::Text {
            role: Role::User,
            content: "What's my name?".to_string(),
        },
    ];

    match client.chat(conversation_messages).await {
        Ok(response) => {
            println!("\n=== Response ===");

            if let Some(usage) = &response.usage {
                if let Some(prompt_tokens) = usage.prompt_tokens {
                    println!("Prompt tokens: {}", prompt_tokens);
                }
                if let Some(completion_tokens) = usage.completion_tokens {
                    println!("Completion tokens: {}", completion_tokens);
                }
            }

            println!("Finish reason: {:?}", response.finish);

            println!("\n=== Messages ===");
            for (i, message) in response.data.iter().enumerate() {
                println!("Message {}: {:?}", i + 1, message);
            }
        }
        Err(e) => {
            eprintln!("Error: {}", e);
        }
    }

    Ok(())
}
