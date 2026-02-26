//! INT4-quantized ONNX embedding model inference.
//!
//! Provides [`EmbeddingModel`] for computing text embeddings using an
//! INT4-quantized ONNX model (e.g. `octen-embedding-0.6b`).
//!
//! # Usage with external weights
//!
//! ```no_run
//! use int4_runner::EmbeddingModel;
//!
//! let tokenizer_json = std::fs::read("path/to/tokenizer.json").unwrap();
//! let model = EmbeddingModel::from_file(
//!     "path/to/model.int4.onnx",
//!     &tokenizer_json,
//! ).unwrap();
//!
//! let embedding = model.embed("hello world").unwrap();
//! println!("embedding dim: {}", embedding.values.len());
//! ```

#[cfg(feature = "server")]
pub mod server;

use ndarray::Array2;
use ort::session::Session;
use ort::value::Value;
use std::path::Path;
use std::sync::Mutex;

const MAX_LENGTH: usize = 512;
const PAD_TOKEN_ID: u32 = 151643;

/// Error type for embedding operations.
#[derive(Debug)]
pub enum Error {
    /// ONNX Runtime error.
    Ort(ort::Error),
    /// Tokenizer loading or encoding error.
    Tokenizer(String),
    /// File I/O error.
    Io(std::io::Error),
    /// Session mutex was poisoned.
    Lock,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Ort(e) => write!(f, "ort: {e}"),
            Error::Tokenizer(e) => write!(f, "tokenizer: {e}"),
            Error::Io(e) => write!(f, "io: {e}"),
            Error::Lock => write!(f, "session lock poisoned"),
        }
    }
}

impl std::error::Error for Error {}

impl From<ort::Error> for Error {
    fn from(e: ort::Error) -> Self {
        Error::Ort(e)
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}

/// A single embedding result.
pub struct Embedding {
    /// The embedding vector.
    pub values: Vec<f32>,
    /// Number of tokens in the input (after truncation).
    pub token_count: u32,
}

/// INT4-quantized ONNX embedding model.
///
/// Thread-safe: multiple threads may call [`embed`](Self::embed) concurrently.
pub struct EmbeddingModel {
    session: Mutex<Session>,
    tokenizer: tokenizers::Tokenizer,
}

impl EmbeddingModel {
    /// Load from an ONNX file on disk and tokenizer JSON bytes.
    ///
    /// `onnx_path` should point to the `.onnx` file. The corresponding
    /// `.onnx.data` file (external weights) must be in the same directory.
    pub fn from_file(
        onnx_path: impl AsRef<Path>,
        tokenizer_json: &[u8],
    ) -> Result<Self, Error> {
        let session = Session::builder()?.commit_from_file(onnx_path.as_ref())?;
        let tokenizer = tokenizers::Tokenizer::from_bytes(tokenizer_json)
            .map_err(|e| Error::Tokenizer(e.to_string()))?;
        Ok(Self {
            session: Mutex::new(session),
            tokenizer,
        })
    }

    /// Load from in-memory ONNX bytes.
    ///
    /// Writes `onnx` and `onnx_data` to a temp directory so ONNX Runtime can
    /// load the external data file, then creates a session.
    pub fn from_bytes(
        onnx: &[u8],
        onnx_data: &[u8],
        tokenizer_json: &[u8],
    ) -> Result<Self, Error> {
        use std::io::Write;
        let dir = std::env::temp_dir().join("int4_runner_embed");
        std::fs::create_dir_all(&dir)?;
        let onnx_path = dir.join("model.int4.onnx");
        let data_path = dir.join("model.int4.onnx.data");
        std::fs::File::create(&onnx_path)?.write_all(onnx)?;
        std::fs::File::create(&data_path)?.write_all(onnx_data)?;
        Self::from_file(onnx_path, tokenizer_json)
    }

    /// Compute the embedding for a single text.
    pub fn embed(&self, text: &str) -> Result<Embedding, Error> {
        let mut session = self.session.lock().map_err(|_| Error::Lock)?;
        compute_embedding(&mut session, &self.tokenizer, text)
    }

    /// Compute embeddings for multiple texts.
    pub fn embed_batch<T: AsRef<str>>(&self, texts: &[T]) -> Result<Vec<Embedding>, Error> {
        let mut session = self.session.lock().map_err(|_| Error::Lock)?;
        texts
            .iter()
            .map(|t| compute_embedding(&mut session, &self.tokenizer, t.as_ref()))
            .collect()
    }
}

fn compute_embedding(
    session: &mut Session,
    tokenizer: &tokenizers::Tokenizer,
    text: &str,
) -> Result<Embedding, Error> {
    let encoding = tokenizer
        .encode(text, true)
        .map_err(|e| Error::Tokenizer(e.to_string()))?;
    let ids = encoding.get_ids();
    let len = ids.len().min(MAX_LENGTH);
    let trunc_ids = if ids.len() > MAX_LENGTH {
        &ids[..MAX_LENGTH]
    } else {
        ids
    };

    let mut input_ids = vec![PAD_TOKEN_ID as i64; MAX_LENGTH];
    let mut attention_mask = vec![0i64; MAX_LENGTH];
    for (i, &id) in trunc_ids.iter().enumerate() {
        input_ids[i] = id as i64;
        attention_mask[i] = 1;
    }

    let input_ids_arr = Array2::from_shape_vec((1, MAX_LENGTH), input_ids)
        .map_err(|e| Error::Tokenizer(e.to_string()))?;
    let attention_mask_arr = Array2::from_shape_vec((1, MAX_LENGTH), attention_mask)
        .map_err(|e| Error::Tokenizer(e.to_string()))?;

    let input_ids_val = Value::from_array(input_ids_arr)?;
    let attention_mask_val = Value::from_array(attention_mask_arr)?;

    let outputs = session.run(ort::inputs![
        "input_ids" => input_ids_val,
        "attention_mask" => attention_mask_val,
    ])?;

    let embeddings = outputs["embeddings"].try_extract_array::<f32>()?;
    let values: Vec<f32> = embeddings.iter().cloned().collect();
    Ok(Embedding {
        values,
        token_count: len as u32,
    })
}
