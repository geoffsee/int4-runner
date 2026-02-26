//! INT4-quantized ONNX embedding model inference.
//!
//! This crate provides [`EmbeddingModel`], a thread-safe wrapper around an
//! INT4-quantized ONNX transformer model for computing dense text embeddings.
//! It is designed for the `octen-embedding-0.6b` model but works with any
//! ONNX model that accepts `input_ids` and `attention_mask` tensors and
//! produces an `embeddings` output tensor.
//!
//! # Architecture
//!
//! ```text
//!   text ──► Tokenizer (BPE) ──► token IDs ──► ONNX Runtime ──► Vec<f32>
//! ```
//!
//! 1. Text is tokenized with a HuggingFace BPE tokenizer (including special tokens).
//! 2. Token sequences are truncated to 512 tokens and padded for batching.
//! 3. The ONNX Runtime performs INT4-quantized inference and returns embedding vectors.
//!
//! # Threading model
//!
//! [`EmbeddingModel`] wraps the ONNX [`Session`] in a
//! [`Mutex`], so it is `Send + Sync` and can be shared
//! across threads via `Arc<EmbeddingModel>`. Only one inference call executes
//! at a time; concurrent callers block on the mutex. The tokenizer is
//! immutable and accessed without locking.
//!
//! # Error handling
//!
//! All fallible operations return `Result<T, Error>`. The [`Error`] enum
//! covers ONNX Runtime failures, tokenizer errors, file I/O problems, and
//! mutex poisoning — no panics in the public API.
//!
//! # Feature flags
//!
//! | Feature  | Default | Description |
//! |----------|---------|-------------|
//! | `server` | **yes** | Enables the [`server`] module with an Axum-based OpenAI-compatible HTTP API. |
//!
//! Disable the server feature to use this crate as a pure embedding library:
//!
//! ```toml
//! [dependencies]
//! int4_runner = { version = "0.2", default-features = false }
//! ```
//!
//! # Loading a model
//!
//! From files on disk (the `.onnx.data` external-weights file must sit next to
//! the `.onnx` file):
//!
//! ```no_run
//! use int4_runner::EmbeddingModel;
//!
//! let tokenizer_json = std::fs::read("tokenizer/tokenizer.json").unwrap();
//! let model = EmbeddingModel::from_file(
//!     "weights/model.int4.onnx",
//!     &tokenizer_json,
//! ).unwrap();
//!
//! let embedding = model.embed("hello world").unwrap();
//! println!("dimensions: {}", embedding.values.len());
//! ```
//!
//! From bytes compiled into the binary (useful for self-contained deployment):
//!
//! ```no_run
//! use int4_runner::EmbeddingModel;
//!
//! static ONNX: &[u8] = include_bytes!("../weights/model.int4.onnx");
//! static DATA: &[u8] = include_bytes!("../weights/model.int4.onnx.data");
//! static TOK:  &[u8] = include_bytes!("../tokenizer/tokenizer.json");
//!
//! let model = EmbeddingModel::from_bytes(ONNX, DATA, TOK).unwrap();
//! ```
//!
//! # Batch inference
//!
//! For multiple texts, [`EmbeddingModel::embed_batch`] is significantly faster
//! than calling [`EmbeddingModel::embed`] in a loop because it performs a single
//! ONNX inference pass over the entire batch:
//!
//! ```no_run
//! # use int4_runner::EmbeddingModel;
//! # let tokenizer_json = std::fs::read("tokenizer/tokenizer.json").unwrap();
//! # let model = EmbeddingModel::from_file("weights/model.int4.onnx", &tokenizer_json).unwrap();
//! let texts = vec!["first sentence", "second sentence", "third sentence"];
//! let embeddings = model.embed_batch(&texts).unwrap();
//! assert_eq!(embeddings.len(), 3);
//! ```

#[cfg(feature = "server")]
pub mod server;

use ndarray::{Array2, Axis};
use ort::session::Session;
use ort::value::Value;
use std::path::Path;
use std::sync::Mutex;

/// Maximum number of tokens per input sequence.
///
/// Sequences longer than this are silently truncated. This limit matches the
/// positional-encoding length of the `octen-embedding-0.6b` model. Inputs
/// shorter than `MAX_LENGTH` are *not* padded when processed individually
/// (see [`EmbeddingModel::embed`]); batch inputs are padded to the longest
/// sequence in the batch (see [`EmbeddingModel::embed_batch`]).
const MAX_LENGTH: usize = 512;

/// Token ID used for padding shorter sequences in batched inference.
///
/// This value (`151643`) is the `<|endoftext|>` token for the Qwen tokenizer
/// used by `octen-embedding-0.6b`. Padded positions are masked out via the
/// attention mask, so the specific ID has no effect on the output.
const PAD_TOKEN_ID: u32 = 151643;

/// Errors that can occur during model loading or inference.
///
/// Every public method on [`EmbeddingModel`] returns `Result<_, Error>`.
/// The variants cover the full surface area: ONNX Runtime failures,
/// tokenizer problems, filesystem I/O, and mutex poisoning.
#[derive(Debug)]
pub enum Error {
    /// The ONNX Runtime returned an error.
    ///
    /// Common causes: missing `.onnx.data` file, incompatible model graph,
    /// or out-of-memory during inference.
    Ort(ort::Error),
    /// The tokenizer could not be loaded from JSON or failed to encode input.
    ///
    /// Wraps the underlying error message as a `String` because the
    /// `tokenizers` crate error type is not `Send + Sync`.
    Tokenizer(String),
    /// A filesystem operation failed (e.g. creating temp files in
    /// [`EmbeddingModel::from_bytes`]).
    Io(std::io::Error),
    /// The internal [`Mutex`] was poisoned by a panicking
    /// thread. Recovery is not possible; the model should be re-created.
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

/// The result of embedding a single text.
///
/// Contains the dense embedding vector produced by the model together with
/// the number of tokens actually consumed (after truncation to 512 tokens).
/// The embedding dimensionality depends on the model; for
/// `octen-embedding-0.6b` it is 1024.
pub struct Embedding {
    /// Dense embedding vector. The length equals the model's hidden
    /// dimension (e.g. 1024 for `octen-embedding-0.6b`).
    pub values: Vec<f32>,
    /// Number of tokens the tokenizer produced for this input, capped at 512.
    /// Useful for tracking token usage and detecting truncation (if
    /// `token_count == 512`, the input was likely truncated).
    pub token_count: u32,
}

/// An INT4-quantized ONNX embedding model ready for inference.
///
/// # Thread safety
///
/// `EmbeddingModel` is `Send + Sync`. The ONNX [`Session`] is protected by a
/// [`Mutex`], so multiple threads may call [`embed`](Self::embed) or
/// [`embed_batch`](Self::embed_batch) concurrently — callers simply block
/// until the lock is available. The [`Tokenizer`](tokenizers::Tokenizer) is
/// immutable after construction and requires no synchronization.
///
/// For maximum throughput under high concurrency, prefer
/// [`embed_batch`](Self::embed_batch) to amortize the lock acquisition over
/// many texts rather than calling [`embed`](Self::embed) per-text.
///
/// # Ownership
///
/// Owns both the ONNX session and tokenizer for the lifetime of the struct.
/// Wrap in [`Arc`](std::sync::Arc) to share across tasks or threads.
pub struct EmbeddingModel {
    /// Mutex-protected ONNX Runtime session. The `ort` `Session` is `!Sync`,
    /// so the mutex is required even though inference is logically read-only.
    session: Mutex<Session>,
    /// BPE tokenizer loaded from HuggingFace-format JSON.
    tokenizer: tokenizers::Tokenizer,
}

impl EmbeddingModel {
    /// Load a model from an ONNX file on disk and tokenizer JSON bytes.
    ///
    /// # Arguments
    ///
    /// * `onnx_path` — Path to the `.onnx` model file. The corresponding
    ///   external-weights file (e.g. `model.int4.onnx.data`) **must** exist in
    ///   the same directory; ONNX Runtime resolves it automatically.
    /// * `tokenizer_json` — Raw bytes of a HuggingFace `tokenizer.json`.
    ///
    /// # Errors
    ///
    /// * [`Error::Ort`] — ONNX Runtime failed to build the session (missing
    ///   file, corrupt graph, unsupported operators, etc.).
    /// * [`Error::Tokenizer`] — The tokenizer JSON could not be parsed.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use int4_runner::EmbeddingModel;
    ///
    /// let tok = std::fs::read("tokenizer/tokenizer.json").unwrap();
    /// let model = EmbeddingModel::from_file("weights/model.int4.onnx", &tok).unwrap();
    /// ```
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

    /// Load a model from in-memory byte slices.
    ///
    /// ONNX Runtime does not support loading external-weight models purely
    /// from memory, so this method writes `onnx` and `onnx_data` to a temp
    /// directory (`{temp_dir}/int4_runner_embed/`) and then delegates to
    /// [`from_file`](Self::from_file). The temp files persist after the call
    /// and are reused on subsequent invocations.
    ///
    /// This is the recommended path for self-contained binaries that embed
    /// the model via [`include_bytes!`].
    ///
    /// # Arguments
    ///
    /// * `onnx` — Raw bytes of the `.onnx` model graph.
    /// * `onnx_data` — Raw bytes of the `.onnx.data` external weights.
    /// * `tokenizer_json` — Raw bytes of a HuggingFace `tokenizer.json`.
    ///
    /// # Errors
    ///
    /// * [`Error::Io`] — Failed to create the temp directory or write files.
    /// * [`Error::Ort`] — ONNX Runtime failed to load the written model.
    /// * [`Error::Tokenizer`] — The tokenizer JSON could not be parsed.
    ///
    /// # Side effects
    ///
    /// Writes two files to `{std::env::temp_dir()}/int4_runner_embed/`:
    /// `model.int4.onnx` and `model.int4.onnx.data`. These are **not**
    /// cleaned up automatically.
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

        // Only write if the files don't exist OR have the wrong size to speed up subsequent startups.
        let onnx_needs_write = std::fs::metadata(&onnx_path).map(|m| m.len()).unwrap_or(0) != onnx.len() as u64;
        if onnx_needs_write {
            std::fs::File::create(&onnx_path)?.write_all(onnx)?;
        }
        let data_needs_write = std::fs::metadata(&data_path).map(|m| m.len()).unwrap_or(0) != onnx_data.len() as u64;
        if data_needs_write {
            std::fs::File::create(&data_path)?.write_all(onnx_data)?;
        }

        Self::from_file(onnx_path, tokenizer_json)
    }

    /// Compute the embedding for a single piece of text.
    ///
    /// Acquires the internal session lock, tokenizes `text`, runs ONNX
    /// inference, and returns the resulting [`Embedding`]. If the input
    /// exceeds 512 tokens it is silently truncated.
    ///
    /// For multiple texts, prefer [`embed_batch`](Self::embed_batch) which
    /// performs a single inference call for the entire batch.
    ///
    /// # Errors
    ///
    /// * [`Error::Lock`] — The session mutex was poisoned.
    /// * [`Error::Tokenizer`] — Tokenization failed.
    /// * [`Error::Ort`] — ONNX inference failed.
    pub fn embed(&self, text: &str) -> Result<Embedding, Error> {
        let mut session = self.session.lock().map_err(|_| Error::Lock)?;
        compute_embedding(&mut session, &self.tokenizer, text)
    }

    /// Compute embeddings for multiple texts in a single batched inference call.
    ///
    /// All inputs are tokenized, padded to the longest sequence in the batch,
    /// and fed to ONNX Runtime as a single `(batch_size, max_len)` tensor pair.
    /// This is substantially faster than calling [`embed`](Self::embed) in a
    /// loop because it amortizes session-lock overhead and enables ONNX
    /// Runtime's internal parallelism across the batch.
    ///
    /// Returns one [`Embedding`] per input text, in the same order. An empty
    /// input slice returns `Ok(vec![])` without acquiring the lock.
    ///
    /// # Type parameter
    ///
    /// `T` can be `String`, `&str`, or any type implementing `AsRef<str>`.
    ///
    /// # Errors
    ///
    /// * [`Error::Lock`] — The session mutex was poisoned.
    /// * [`Error::Tokenizer`] — Tokenization failed for any input text.
    /// * [`Error::Ort`] — ONNX inference failed.
    pub fn embed_batch<T: AsRef<str>>(&self, texts: &[T]) -> Result<Vec<Embedding>, Error> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let mut session = self.session.lock().map_err(|_| Error::Lock)?;
        let text_refs: Vec<&str> = texts.iter().map(|t| t.as_ref()).collect();
        compute_embeddings_batch(&mut session, &self.tokenizer, &text_refs)
    }
}

/// Tokenize a single text, run inference, and return its embedding.
///
/// This is the inner implementation behind [`EmbeddingModel::embed`]. The
/// caller is responsible for holding the session lock.
///
/// # Pipeline
///
/// 1. BPE-encode `text` with special tokens.
/// 2. Truncate to [`MAX_LENGTH`] tokens.
/// 3. Build `(1, len)` shaped `input_ids` and `attention_mask` tensors.
/// 4. Run the ONNX session and extract the `"embeddings"` output.
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

    let input_ids: Vec<i64> = trunc_ids.iter().map(|&id| id as i64).collect();
    let attention_mask = vec![1i64; len];

    let input_ids_arr = Array2::from_shape_vec((1, len), input_ids)
        .map_err(|e| Error::Tokenizer(e.to_string()))?;
    let attention_mask_arr = Array2::from_shape_vec((1, len), attention_mask)
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

/// Tokenize multiple texts, build a padded batch, and run a single inference call.
///
/// This is the inner implementation behind [`EmbeddingModel::embed_batch`].
/// The caller is responsible for holding the session lock.
///
/// # Pipeline
///
/// 1. BPE-encode each text with special tokens and truncate to [`MAX_LENGTH`].
/// 2. Pad all sequences to the longest in the batch using [`PAD_TOKEN_ID`].
/// 3. Build `(batch_size, max_len)` shaped `input_ids` and `attention_mask` tensors.
/// 4. Run the ONNX session once.
/// 5. Slice the `"embeddings"` output along axis 0 to extract per-text vectors.
fn compute_embeddings_batch(
    session: &mut Session,
    tokenizer: &tokenizers::Tokenizer,
    texts: &[&str],
) -> Result<Vec<Embedding>, Error> {
    let n = texts.len();

    let mut all_ids: Vec<Vec<i64>> = Vec::with_capacity(n);
    let mut token_counts: Vec<u32> = Vec::with_capacity(n);
    let mut max_len: usize = 0;

    for &text in texts {
        let encoding = tokenizer
            .encode(text, true)
            .map_err(|e| Error::Tokenizer(e.to_string()))?;
        let ids = encoding.get_ids();
        let len = ids.len().min(MAX_LENGTH);
        let trunc_ids: Vec<i64> = ids[..len].iter().map(|&id| id as i64).collect();
        max_len = max_len.max(len);
        token_counts.push(len as u32);
        all_ids.push(trunc_ids);
    }

    let mut input_ids = vec![PAD_TOKEN_ID as i64; n * max_len];
    let mut attention_mask = vec![0i64; n * max_len];

    for (i, ids) in all_ids.iter().enumerate() {
        let offset = i * max_len;
        for (j, &id) in ids.iter().enumerate() {
            input_ids[offset + j] = id;
            attention_mask[offset + j] = 1;
        }
    }

    let input_ids_arr = Array2::from_shape_vec((n, max_len), input_ids)
        .map_err(|e| Error::Tokenizer(e.to_string()))?;
    let attention_mask_arr = Array2::from_shape_vec((n, max_len), attention_mask)
        .map_err(|e| Error::Tokenizer(e.to_string()))?;

    let input_ids_val = Value::from_array(input_ids_arr)?;
    let attention_mask_val = Value::from_array(attention_mask_arr)?;

    let outputs = session.run(ort::inputs![
        "input_ids" => input_ids_val,
        "attention_mask" => attention_mask_val,
    ])?;

    let embeddings = outputs["embeddings"].try_extract_array::<f32>()?;

    let results: Vec<Embedding> = (0..n)
        .map(|i| {
            let row = embeddings.index_axis(Axis(0), i);
            Embedding {
                values: row.iter().cloned().collect(),
                token_count: token_counts[i],
            }
        })
        .collect();

    Ok(results)
}
