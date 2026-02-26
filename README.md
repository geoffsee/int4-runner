# Qwen3 ONNX INT4

Qwen3 ONNX embeddings server (experimental)

## Run
1. Download an artifact from a workflow
2. Do *things* with it
```bash
./int4_runner
# Embeddings API: http://0.0.0.0:8080/v1/embeddings
```

### Optional env

- **`PORT`** – Listen port (default: `8080`).

## API: OpenAI-compatible embeddings

**POST** `/v1/embeddings`

- **Request:** JSON body with `input` (string or array of strings) and optional `model`.
- **Response:** Same shape as OpenAI’s embeddings API (`data[].embedding`, `usage`, etc.).

Example:

```bash
curl -X POST http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": "Your text here"}'
```

CORS is enabled so the endpoint can be called from browsers.

## License

MIT 2026 Copyright Geoff Seemueller