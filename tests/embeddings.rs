use int4_runner::EmbeddingModel;
use std::sync::{Arc, OnceLock};

/// Single shared model instance for all tests in this binary.
fn model() -> &'static Arc<EmbeddingModel> {
    static MODEL: OnceLock<Arc<EmbeddingModel>> = OnceLock::new();
    MODEL.get_or_init(|| {
        let tok = std::fs::read("tokenizer/tokenizer.json").expect("tokenizer.json not found");
        let m = EmbeddingModel::from_file("weights/model.int4.onnx", &tok)
            .expect("failed to load model");
        Arc::new(m)
    })
}

/// Cosine similarity between two vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

// ── Golden / Snapshot Tests ─────────────────────────────────────────────

mod golden {
    use super::*;
    use serde::{Deserialize, Serialize};

    #[derive(Serialize, Deserialize)]
    struct GoldenEntry {
        input: String,
        embedding: Vec<f32>,
    }

    #[test]
    fn golden_embeddings_match_fixture() {
        let path = "tests/fixtures/golden_embeddings.json";
        let data = std::fs::read_to_string(path)
            .unwrap_or_else(|_| panic!("Golden fixture not found at {path}. Run: cargo test -- --ignored generate_golden_fixture"));
        let entries: Vec<GoldenEntry> = serde_json::from_str(&data).unwrap();
        let model = model();

        for entry in &entries {
            let emb = model.embed(&entry.input).expect("embed failed");
            assert_eq!(
                emb.values.len(),
                entry.embedding.len(),
                "dimension mismatch for {:?}",
                entry.input
            );
            for (i, (got, want)) in emb.values.iter().zip(&entry.embedding).enumerate() {
                assert!(
                    (got - want).abs() < 1e-5,
                    "mismatch at dim {i} for {:?}: got {got}, want {want}",
                    entry.input
                );
            }
        }
    }

    #[test]
    #[ignore]
    fn generate_golden_fixture() {
        let model = model();
        let inputs = [
            "hello world",
            "The quick brown fox jumps over the lazy dog",
            "Rust is a systems programming language",
            "machine learning",
            "quantum computing",
        ];
        let entries: Vec<GoldenEntry> = inputs
            .iter()
            .map(|&input| {
                let emb = model.embed(input).expect("embed failed");
                GoldenEntry {
                    input: input.to_string(),
                    embedding: emb.values,
                }
            })
            .collect();

        let json = serde_json::to_string_pretty(&entries).unwrap();
        std::fs::write("tests/fixtures/golden_embeddings.json", json).unwrap();
        println!("Wrote golden fixture with {} entries", entries.len());
    }
}

// ── Structural / Invariant Tests ────────────────────────────────────────

mod structural {
    use super::*;

    #[test]
    fn embedding_has_correct_dimension() {
        let emb = model().embed("test input").unwrap();
        assert_eq!(emb.values.len(), 1024);
    }

    #[test]
    fn embedding_values_are_finite() {
        let emb = model().embed("check for NaN or Inf").unwrap();
        for (i, &v) in emb.values.iter().enumerate() {
            assert!(v.is_finite(), "non-finite value at dim {i}: {v}");
        }
    }

    #[test]
    fn embedding_is_nonzero() {
        let emb = model().embed("nonzero check").unwrap();
        let norm: f32 = emb.values.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(norm > 0.0, "embedding has zero L2 norm");
    }

    #[test]
    fn deterministic_same_input() {
        let model = model();
        let a = model.embed("deterministic test").unwrap();
        let b = model.embed("deterministic test").unwrap();
        for (i, (va, vb)) in a.values.iter().zip(&b.values).enumerate() {
            assert!(
                (va - vb).abs() < 1e-7,
                "non-deterministic at dim {i}: {va} vs {vb}"
            );
        }
    }

    #[test]
    fn batch_single_equivalence() {
        let model = model();
        let single = model.embed("equivalence").unwrap();
        let batch = model.embed_batch(&["equivalence"]).unwrap();
        assert_eq!(batch.len(), 1);
        for (i, (s, b)) in single.values.iter().zip(&batch[0].values).enumerate() {
            assert!(
                (s - b).abs() < 1e-5,
                "batch/single mismatch at dim {i}: {s} vs {b}"
            );
        }
    }

    #[test]
    fn batch_preserves_order() {
        let model = model();
        let texts = ["alpha", "beta", "gamma"];
        let batch = model.embed_batch(&texts).unwrap();
        assert_eq!(batch.len(), 3);
        for (idx, text) in texts.iter().enumerate() {
            let single = model.embed(text).unwrap();
            let sim = cosine_similarity(&single.values, &batch[idx].values);
            assert!(
                sim > 0.999,
                "order mismatch for {text:?} at index {idx}: sim = {sim}"
            );
        }
    }

    #[test]
    fn empty_batch_returns_empty() {
        let result = model().embed_batch::<&str>(&[]).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn single_char_input() {
        let emb = model().embed("a").unwrap();
        assert_eq!(emb.values.len(), 1024);
    }

    #[test]
    fn token_count_is_positive() {
        let emb = model().embed("some text").unwrap();
        assert!(emb.token_count >= 1, "token count should be >= 1");
    }

    #[test]
    fn long_input_truncation() {
        let long_text = "word ".repeat(2000);
        let emb = model().embed(&long_text).unwrap();
        assert!(
            emb.token_count <= 512,
            "token count {} exceeds 512",
            emb.token_count
        );
    }

    #[test]
    fn token_count_increases_with_length() {
        let model = model();
        let short = model.embed("hi").unwrap();
        let long = model.embed("This is a significantly longer sentence with many more words in it").unwrap();
        assert!(
            long.token_count > short.token_count,
            "longer text should have more tokens: {} vs {}",
            long.token_count,
            short.token_count
        );
    }

    #[test]
    fn batch_of_one_equals_single() {
        let model = model();
        let text = "batch of one";
        let single = model.embed(text).unwrap();
        let batch = model.embed_batch(&[text]).unwrap();
        assert_eq!(single.values.len(), batch[0].values.len());
        for (i, (s, b)) in single.values.iter().zip(&batch[0].values).enumerate() {
            assert!(
                (s - b).abs() < 1e-5,
                "mismatch at dim {i}: {s} vs {b}"
            );
        }
    }
}

// ── Semantic Quality Tests ──────────────────────────────────────────────

mod semantic {
    use super::*;

    #[test]
    fn related_concepts_more_similar_than_unrelated() {
        let model = model();
        let dog = model.embed("dog").unwrap();
        let puppy = model.embed("puppy").unwrap();
        let quantum = model.embed("quantum physics").unwrap();

        let sim_related = cosine_similarity(&dog.values, &puppy.values);
        let sim_unrelated = cosine_similarity(&dog.values, &quantum.values);
        assert!(
            sim_related > sim_unrelated,
            "dog-puppy ({sim_related}) should be more similar than dog-quantum ({sim_unrelated})"
        );
    }

    #[test]
    fn identical_inputs_have_similarity_one() {
        let emb = model().embed("identical").unwrap();
        let sim = cosine_similarity(&emb.values, &emb.values);
        assert!(
            (sim - 1.0).abs() < 1e-5,
            "self-similarity should be ~1.0, got {sim}"
        );
    }

    #[test]
    fn semantic_similarity_pairs() {
        let model = model();
        let pairs = [
            ("king", "queen"),
            ("cat", "kitten"),
            ("happy", "joyful"),
            ("car", "automobile"),
            ("computer", "laptop"),
        ];
        for (a_text, b_text) in &pairs {
            let a = model.embed(a_text).unwrap();
            let b = model.embed(b_text).unwrap();
            let sim = cosine_similarity(&a.values, &b.values);
            assert!(
                sim > 0.5,
                "{a_text}-{b_text} similarity {sim} should be > 0.5"
            );
        }
    }

    #[test]
    fn unrelated_concepts_have_low_similarity() {
        let model = model();
        let pairs = [
            ("banana", "differential equations"),
            ("sunset", "database schema"),
            ("guitar", "mitochondria"),
        ];
        for (a_text, b_text) in &pairs {
            let a = model.embed(a_text).unwrap();
            let b = model.embed(b_text).unwrap();
            let sim = cosine_similarity(&a.values, &b.values);
            assert!(
                sim < 0.5,
                "{a_text}-{b_text} similarity {sim} should be < 0.5"
            );
        }
    }

    #[test]
    fn different_inputs_produce_different_embeddings() {
        let model = model();
        let a = model.embed("The weather is sunny today").unwrap();
        let b = model.embed("Quantum entanglement in superconductors").unwrap();
        let sim = cosine_similarity(&a.values, &b.values);
        assert!(
            sim < 0.95,
            "dissimilar texts should have sim < 0.95, got {sim}"
        );
    }
}

// ── HTTP Server Tests ───────────────────────────────────────────────────

mod server {
    use super::*;
    use axum::body::Body;
    use http_body_util::BodyExt;
    use int4_runner::server::app;
    use tower::ServiceExt;

    fn test_app() -> axum::Router {
        app(model().clone())
    }

    async fn post_embeddings(body: &str) -> (u16, serde_json::Value) {
        let app = test_app();
        let req = axum::http::Request::builder()
            .method("POST")
            .uri("/v1/embeddings")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        let status = resp.status().as_u16();
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap_or_default();
        (status, json)
    }

    #[tokio::test]
    async fn single_string_input() {
        let (status, json) = post_embeddings(r#"{"input":"hello"}"#).await;
        assert_eq!(status, 200);
        let data = json["data"].as_array().unwrap();
        assert_eq!(data.len(), 1);
    }

    #[tokio::test]
    async fn batch_array_input() {
        let (status, json) = post_embeddings(r#"{"input":["a","b","c"]}"#).await;
        assert_eq!(status, 200);
        let data = json["data"].as_array().unwrap();
        assert_eq!(data.len(), 3);
    }

    #[tokio::test]
    async fn response_shape_matches_openai() {
        let (status, json) = post_embeddings(r#"{"input":"test"}"#).await;
        assert_eq!(status, 200);
        assert_eq!(json["object"].as_str().unwrap(), "list");
        assert!(json["data"].is_array());
        assert!(json["model"].is_string());
        assert!(json["usage"]["prompt_tokens"].is_u64());
        assert!(json["usage"]["total_tokens"].is_u64());

        let item = &json["data"][0];
        assert_eq!(item["object"].as_str().unwrap(), "embedding");
        assert!(item["index"].is_u64());
        assert!(item["embedding"].is_array());
    }

    #[tokio::test]
    async fn custom_model_name_echoed() {
        let (_, json) =
            post_embeddings(r#"{"input":"test","model":"my-model"}"#).await;
        assert_eq!(json["model"].as_str().unwrap(), "my-model");
    }

    #[tokio::test]
    async fn default_model_name() {
        let (_, json) = post_embeddings(r#"{"input":"test"}"#).await;
        assert_eq!(json["model"].as_str().unwrap(), "octen-embedding-0.6b");
    }

    #[tokio::test]
    async fn texts_alias_for_input() {
        let (status, json) = post_embeddings(r#"{"texts":"hello"}"#).await;
        assert_eq!(status, 200);
        let data = json["data"].as_array().unwrap();
        assert_eq!(data.len(), 1);
    }

    #[tokio::test]
    async fn empty_array_returns_400() {
        let (status, _) = post_embeddings(r#"{"input":[]}"#).await;
        assert_eq!(status, 400);
    }

    #[tokio::test]
    async fn invalid_json_returns_error() {
        let (status, _) = post_embeddings("not json at all").await;
        assert!(status >= 400 && status < 500, "expected 4xx, got {status}");
    }

    #[tokio::test]
    async fn wrong_http_method_returns_405() {
        let app = test_app();
        let req = axum::http::Request::builder()
            .method("GET")
            .uri("/v1/embeddings")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status().as_u16(), 405);
    }

    #[tokio::test]
    async fn wrong_path_returns_404() {
        let app = test_app();
        let req = axum::http::Request::builder()
            .method("POST")
            .uri("/v1/nonexistent")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"input":"test"}"#.to_string()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status().as_u16(), 404);
    }

    #[tokio::test]
    async fn usage_token_count_correct() {
        let (_, json) = post_embeddings(r#"{"input":"hello world"}"#).await;
        let prompt = json["usage"]["prompt_tokens"].as_u64().unwrap();
        let total = json["usage"]["total_tokens"].as_u64().unwrap();
        assert_eq!(prompt, total);
        assert!(prompt > 0);
    }

    #[tokio::test]
    async fn embedding_dimension_in_response() {
        let (_, json) = post_embeddings(r#"{"input":"test"}"#).await;
        let embedding = json["data"][0]["embedding"].as_array().unwrap();
        assert_eq!(embedding.len(), 1024);
    }

    #[tokio::test]
    async fn batch_indices_are_sequential() {
        let (_, json) = post_embeddings(r#"{"input":["a","b","c"]}"#).await;
        let data = json["data"].as_array().unwrap();
        for (i, item) in data.iter().enumerate() {
            assert_eq!(item["index"].as_u64().unwrap(), i as u64);
        }
    }

    #[tokio::test]
    async fn cors_headers_present() {
        let app = test_app();
        let req = axum::http::Request::builder()
            .method("OPTIONS")
            .uri("/v1/embeddings")
            .header("origin", "http://example.com")
            .header("access-control-request-method", "POST")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert!(
            resp.headers().contains_key("access-control-allow-origin"),
            "missing CORS allow-origin header"
        );
    }
}

// ── Concurrency Tests ───────────────────────────────────────────────────

mod concurrency {
    use super::*;

    #[test]
    fn concurrent_embed_calls() {
        let model = model().clone();
        let handles: Vec<_> = (0..8)
            .map(|i| {
                let m = model.clone();
                std::thread::spawn(move || {
                    m.embed(&format!("concurrent text {i}"))
                        .expect("embed failed in thread")
                })
            })
            .collect();
        for h in handles {
            let emb = h.join().expect("thread panicked");
            assert_eq!(emb.values.len(), 1024);
        }
    }

    #[test]
    fn concurrent_embed_deterministic() {
        let model = model().clone();
        let text = "deterministic concurrent";
        let handles: Vec<_> = (0..4)
            .map(|_| {
                let m = model.clone();
                let t = text.to_string();
                std::thread::spawn(move || m.embed(&t).expect("embed failed"))
            })
            .collect();
        let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        for i in 1..results.len() {
            for (d, (a, b)) in results[0].values.iter().zip(&results[i].values).enumerate() {
                assert!(
                    (a - b).abs() < 1e-7,
                    "thread {i} diverged at dim {d}: {a} vs {b}"
                );
            }
        }
    }
}

// ── Performance Tests (ignored by default) ──────────────────────────────

mod performance {
    use super::*;
    use std::time::Instant;

    #[test]
    #[ignore]
    fn single_embed_latency() {
        let model = model();
        // Warmup
        let _ = model.embed("warmup").unwrap();

        let iterations = 20;
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = model.embed("benchmark single embedding latency").unwrap();
        }
        let elapsed = start.elapsed();
        let avg_ms = elapsed.as_millis() as f64 / iterations as f64;
        println!("single_embed_latency: {avg_ms:.2} ms avg over {iterations} iterations");
    }

    #[test]
    #[ignore]
    fn batch_embed_latency() {
        let model = model();
        let texts: Vec<String> = (0..16).map(|i| format!("batch benchmark text number {i}")).collect();
        let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();

        // Warmup
        let _ = model.embed_batch(&refs).unwrap();

        let iterations = 10;
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = model.embed_batch(&refs).unwrap();
        }
        let elapsed = start.elapsed();
        let avg_ms = elapsed.as_millis() as f64 / iterations as f64;
        let per_text_ms = avg_ms / texts.len() as f64;
        println!(
            "batch_embed_latency: {avg_ms:.2} ms avg ({per_text_ms:.2} ms/text) over {iterations} iterations, batch_size={}",
            texts.len()
        );
    }
}
