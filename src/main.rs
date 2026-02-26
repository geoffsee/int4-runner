//! Self-contained embedding server binary.
//!
//! This binary embeds the full INT4-quantized ONNX model (`~4 MB` graph +
//! `~533 MB` external weights) and the BPE tokenizer (`~11 MB`) directly into
//! the executable via [`include_bytes!`]. No external files are needed at
//! runtime — just run the binary.
//!
//! On startup, the model bytes are written to a temp directory (required by
//! ONNX Runtime for external-weight models), the [`EmbeddingModel`] is
//! initialized, and an OpenAI-compatible HTTP server is started on the
//! configured port.
//!
//! # Configuration
//!
//! | Variable | Default | Description |
//! |----------|---------|-------------|
//! | `PORT`   | `8080`  | TCP port for the HTTP server |
//!
//! # Example
//!
//! ```bash
//! $ PORT=3000 ./int4_runner
//! Model loaded from embedded bytes
//! Embeddings API: http://0.0.0.0:3000/v1/embeddings
//! ```

use int4_runner::EmbeddingModel;
use int4_runner::server::run_server;

/// INT4-quantized ONNX model graph (~4 MB).
static MODEL_ONNX: &[u8] = include_bytes!("../weights/model.int4.onnx");
/// External weight data for the ONNX model (~533 MB).
static MODEL_ONNX_DATA: &[u8] = include_bytes!("../weights/model.int4.onnx.data");
/// HuggingFace BPE tokenizer configuration (~11 MB).
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
