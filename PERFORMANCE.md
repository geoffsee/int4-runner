# Performance Optimization

If inference is slower than expected, here are several strategies to optimize your ONNX Runtime performance with `int4-runner`.

### 1. Graph Simplification with `onnxsim`
Direct graph exports from frameworks like PyTorch often contain redundant or "junk" nodes that can hinder performance. [`onnx-simplifier` (onnxsim)](https://github.com/daquexian/onnx-simplifier) is a powerful tool that simplifies the ONNX graph by evaluating constant nodes and folding them, which can significantly reduce latency.

```bash
pip install onnxsim
onnxsim model.onnx model_sim.onnx
```

### 2. Select the Right Execution Provider (EP)
By default, ONNX Runtime uses the CPU. To leverage hardware acceleration, you should enable and configure the appropriate Execution Provider for your platform:
*   **NVIDIA GPUs**: Use the [`CUDA`](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html) or [`TensorRT`](https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html) EPs.
*   **Apple Silicon**: Use the [`CoreML`](https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html) EP.
*   **Windows (DirectX 12)**: Use the [`DirectML`](https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html) EP.
*   **Intel Hardware**: Use the [`OpenVINO`](https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html) EP.

> **Note:** Enabling these in `ort` requires adding the corresponding feature flags to your `Cargo.toml`.

### 3. Use the Transformer Optimization Tool
For transformer-based models (like BERT, GPT, or Qwen), use the [`onnxruntime.transformers`](https://github.com/microsoft/onnxruntime/tree/main/onnxruntime/python/tools/transformers) optimization script. This tool performs advanced operator fusion and graph transformations (e.g., merging LayerNorms and attention heads) specifically tailored for transformer architectures, often resulting in much more efficient graphs.

### 4. Implement I/O Binding
When using hardware accelerators (like GPUs), copying input/output tensors between the CPU and the device can be a major bottleneck. **I/O Binding** allows you to provide pre-allocated device memory directly to the session, eliminating redundant data copies and reducing the overall latency of the inference pipeline.

### 5. Model Quantization & Mixed Precision
*   **INT4 Quantization**: This project focuses on 4-bit quantization, which offers massive speedups and reduced memory bandwidth requirements compared to FP32 or even FP16, especially on edge devices.
*   **8-bit Quantization (INT8)**: A common alternative that provides a large performance boost with minimal accuracy loss.
*   **Mixed Precision (FP16)**: If accuracy loss from quantization is too high, using half-precision (Float16) can still offer significant speedups on modern hardware (like NVIDIA Tensor Cores or Apple ANE) while maintaining high precision.

### 6. Configure Threading (Intra-op/Inter-op)
The default threading strategy might not be optimal for all workloads. You can manually tune the number of threads ONNX Runtime uses:
*   **Intra-op Threads**: Number of threads used to parallelize a single operator.
*   **Inter-op Threads**: Number of threads used to parallelize multiple independent operators (best for complex, multi-branch graphs).

In `ort`, you can configure these via the `SessionBuilder`:
```rust
let session = Session::builder()?
    .with_intra_threads(4)?
    .commit_from_file("model.onnx")?;
```

### 7. Batching
For multiple texts, `EmbeddingModel::embed_batch` is substantially faster than calling `EmbeddingModel::embed` in a loop because it amortizes session-lock overhead and enables internal parallelism across the batch.
