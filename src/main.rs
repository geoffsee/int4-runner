//! Embed text with the INT4 ONNX Octen model.
//!
//! Model and tokenizer bytes are compiled into the binary; no external files at runtime.
//! Serves an OpenAI-compatible `/v1/embeddings` HTTP API.

use int4_runner::EmbeddingModel;
use int4_runner::server::run_server;

static MODEL_ONNX: &[u8] = include_bytes!("../weights/model.int4.onnx");
static MODEL_ONNX_DATA: &[u8] = include_bytes!("../weights/model.int4.onnx.data");
static TOKENIZER_JSON: &[u8] = include_bytes!("../tokenizer/tokenizer.json");

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let model = EmbeddingModel::from_bytes(MODEL_ONNX, MODEL_ONNX_DATA, TOKENIZER_JSON)?;
    println!("Model loaded from embedded bytes");

    let port: u16 = std::env::var("PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(8080);

    run_server(model, port).await
}
