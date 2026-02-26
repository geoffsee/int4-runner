//! OpenAI-compatible embeddings HTTP API.

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

const MODEL_ID: &str = "octen-embedding-0.6b";

/// OpenAI-compatible request body. Accepts "input" (OpenAI) or "texts" (array of strings).
#[derive(Debug, Deserialize)]
pub struct EmbeddingsRequest {
    #[serde(default)]
    pub model: Option<String>,
    /// OpenAI uses "input"; some clients send "texts". We accept both.
    #[serde(alias = "texts")]
    pub input: EmbeddingInput,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingInput {
    Single(String),
    Batch(Vec<String>),
}

impl EmbeddingInput {
    fn texts(self) -> Vec<String> {
        match self {
            EmbeddingInput::Single(s) => vec![s],
            EmbeddingInput::Batch(v) => v,
        }
    }
}

/// OpenAI-compatible response.
#[derive(Debug, Serialize)]
pub struct EmbeddingsResponse {
    pub object: &'static str,
    pub data: Vec<EmbeddingItem>,
    pub model: String,
    pub usage: Usage,
}

#[derive(Debug, Serialize)]
pub struct EmbeddingItem {
    pub object: &'static str,
    pub index: u32,
    pub embedding: Vec<f32>,
}

#[derive(Debug, Serialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub total_tokens: u32,
}

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

    let mut data = Vec::with_capacity(texts.len());
    let mut total_tokens = 0u32;

    for (index, text) in texts.iter().enumerate() {
        let emb = model
            .embed(text)
            .map_err(|e| (StatusCode::UNPROCESSABLE_ENTITY, e.to_string()))?;

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
