//! OpenAI-compatible embeddings HTTP API.
//!
//! This module provides [`run_server`], which starts an Axum HTTP server
//! exposing a single endpoint:
//!
//! ```text
//! POST /v1/embeddings
//! ```
//!
//! The request and response shapes match the
//! [OpenAI Embeddings API](https://platform.openai.com/docs/api/embeddings),
//! so this server can be used as a drop-in local replacement for
//! `https://api.openai.com/v1/embeddings` in any client that supports a
//! configurable base URL.
//!
//! # Request format
//!
//! ```json
//! {
//!   "input": "Some text to embed",
//!   "model": "octen-embedding-0.6b"   // optional
//! }
//! ```
//!
//! `input` can be a single string or an array of strings. The field name
//! `"texts"` is accepted as an alias for `"input"` for compatibility with
//! non-OpenAI clients.
//!
//! # Response format
//!
//! ```json
//! {
//!   "object": "list",
//!   "data": [
//!     { "object": "embedding", "index": 0, "embedding": [0.012, ...] }
//!   ],
//!   "model": "octen-embedding-0.6b",
//!   "usage": { "prompt_tokens": 5, "total_tokens": 5 }
//! }
//! ```
//!
//! # Error responses
//!
//! | Status | Condition |
//! |--------|-----------|
//! | `400`  | Empty input string or empty array |
//! | `422`  | Tokenization or ONNX inference failure |
//!
//! # CORS
//!
//! All origins, methods, and headers are allowed, so the endpoint can be
//! called directly from browser JavaScript.
//!
//! # Concurrency
//!
//! The [`EmbeddingModel`] is wrapped in an [`Arc`] and shared across all
//! Axum handler tasks. Concurrent requests contend on the model's internal
//! [`Mutex`](std::sync::Mutex); see [`EmbeddingModel`] docs for details.

use crate::EmbeddingModel;
use axum::{
    extract::State,
    http::StatusCode,
    routing::post,
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tower_http::cors::{Any, CorsLayer};

/// Default model identifier returned in API responses when the request does
/// not specify a `model` field.
const MODEL_ID: &str = "octen-embedding-0.6b";

/// OpenAI-compatible embeddings request body.
///
/// Accepts the standard `"input"` field (per the OpenAI spec) as well as
/// `"texts"` as an alias for compatibility with alternative clients.
///
/// # JSON examples
///
/// Single string:
/// ```json
/// { "input": "hello world" }
/// ```
///
/// Batch:
/// ```json
/// { "input": ["hello", "world"], "model": "custom-model" }
/// ```
#[derive(Debug, Deserialize)]
pub struct EmbeddingsRequest {
    /// Optional model name echoed back in the response. Defaults to
    /// `"octen-embedding-0.6b"` if omitted.
    #[serde(default)]
    pub model: Option<String>,
    /// The text(s) to embed. Deserialized from either `"input"` or `"texts"`.
    #[serde(alias = "texts")]
    pub input: EmbeddingInput,
}

/// The `"input"` field of an embeddings request.
///
/// Uses `#[serde(untagged)]` so a JSON string deserializes as [`Single`](Self::Single)
/// and a JSON array of strings deserializes as [`Batch`](Self::Batch).
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingInput {
    /// A single text string.
    Single(String),
    /// An array of text strings.
    Batch(Vec<String>),
}

impl EmbeddingInput {
    /// Normalize into a `Vec<String>` regardless of variant.
    fn texts(self) -> Vec<String> {
        match self {
            EmbeddingInput::Single(s) => vec![s],
            EmbeddingInput::Batch(v) => v,
        }
    }
}

/// OpenAI-compatible embeddings response.
///
/// Serializes to the same JSON shape as the OpenAI
/// [`/v1/embeddings`](https://platform.openai.com/docs/api/embeddings)
/// endpoint, so existing clients can consume it without modification.
#[derive(Debug, Serialize)]
pub struct EmbeddingsResponse {
    /// Always `"list"`.
    pub object: &'static str,
    /// One [`EmbeddingItem`] per input text, in order.
    pub data: Vec<EmbeddingItem>,
    /// Model name (from the request, or `"octen-embedding-0.6b"` by default).
    pub model: String,
    /// Aggregate token usage across all inputs in the batch.
    pub usage: Usage,
}

/// A single embedding within the response `data` array.
#[derive(Debug, Serialize)]
pub struct EmbeddingItem {
    /// Always `"embedding"`.
    pub object: &'static str,
    /// Zero-based position in the batch corresponding to the input order.
    pub index: u32,
    /// Dense embedding vector (length = model hidden dimension).
    pub embedding: Vec<f32>,
}

/// Token usage statistics for the request.
#[derive(Debug, Serialize)]
pub struct Usage {
    /// Total tokens consumed across all input texts (after truncation).
    pub prompt_tokens: u32,
    /// Equal to `prompt_tokens` (embeddings have no completion tokens).
    pub total_tokens: u32,
}

/// Axum handler for `POST /v1/embeddings`.
///
/// Validates that the input is non-empty, delegates to
/// [`EmbeddingModel::embed_batch`], and assembles an OpenAI-compatible
/// JSON response. Returns `400` for empty input and `422` for inference
/// errors.
async fn embeddings_handler(
    State(model): State<Arc<EmbeddingModel>>,
    Json(req): Json<EmbeddingsRequest>,
) -> Result<Json<EmbeddingsResponse>, (StatusCode, String)> {
    let texts = req.input.texts();
    if texts.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            "input must be a non-empty string or array of strings".to_string(),
        ));
    }

    let results = model
        .embed_batch(&texts)
        .map_err(|e| (StatusCode::UNPROCESSABLE_ENTITY, e.to_string()))?;

    let mut total_tokens = 0u32;
    let mut data = Vec::with_capacity(results.len());

    for (index, emb) in results.into_iter().enumerate() {
        total_tokens += emb.token_count;
        data.push(EmbeddingItem {
            object: "embedding",
            index: index as u32,
            embedding: emb.values,
        });
    }

    let model_name = req.model.unwrap_or_else(|| MODEL_ID.to_string());

    Ok(Json(EmbeddingsResponse {
        object: "list",
        data,
        model: model_name,
        usage: Usage {
            prompt_tokens: total_tokens,
            total_tokens,
        },
    }))
}

/// Start the embeddings HTTP server.
///
/// Binds to `0.0.0.0:{port}` and serves an OpenAI-compatible
/// `POST /v1/embeddings` endpoint until the process is terminated.
///
/// # Arguments
///
/// * `model` — A fully initialized [`EmbeddingModel`]. Ownership is moved
///   into an [`Arc`] for shared access across handler tasks.
/// * `port` — TCP port to listen on (e.g. `8080`).
///
/// # Errors
///
/// Returns an error if the TCP listener fails to bind (e.g. port already
/// in use) or the server encounters a fatal I/O error.
///
/// # Example
///
/// ```no_run
/// use int4_runner::EmbeddingModel;
/// use int4_runner::server::run_server;
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
/// let tok = std::fs::read("tokenizer/tokenizer.json")?;
/// let model = EmbeddingModel::from_file("weights/model.int4.onnx", &tok)?;
/// run_server(model, 8080).await?;
/// # Ok(())
/// # }
/// ```
pub async fn run_server(
    model: EmbeddingModel,
    port: u16,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let state = Arc::new(model);

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = Router::new()
        .route("/v1/embeddings", post(embeddings_handler))
        .layer(cors)
        .with_state(state);

    let addr = (std::net::Ipv4Addr::UNSPECIFIED, port);
    let listener = tokio::net::TcpListener::bind(addr).await?;
    println!("Embeddings API: http://0.0.0.0:{port}/v1/embeddings");
    axum::serve(listener, app).await?;
    Ok(())
}
