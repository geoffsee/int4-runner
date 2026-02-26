use int4_runner::EmbeddingModel;
use std::fs;
use std::io::Write;

#[test]
fn test_from_bytes_writes_files_and_loads() {
    let onnx = include_bytes!("../weights/model.int4.onnx");
    let onnx_data = include_bytes!("../weights/model.int4.onnx.data");
    let tok = include_bytes!("../tokenizer/tokenizer.json");

    let dir = std::env::temp_dir().join("int4_runner_embed");
    let onnx_path = dir.join("model.int4.onnx");
    let data_path = dir.join("model.int4.onnx.data");

    // Clear current state to ensure we test write logic.
    if onnx_path.exists() { fs::remove_file(&onnx_path).unwrap(); }
    if data_path.exists() { fs::remove_file(&data_path).unwrap(); }

    let model = EmbeddingModel::from_bytes(onnx, onnx_data, tok).expect("Failed to load from bytes");
    
    assert!(onnx_path.exists());
    assert!(data_path.exists());
    assert_eq!(fs::metadata(&onnx_path).unwrap().len(), onnx.len() as u64);
    assert_eq!(fs::metadata(&data_path).unwrap().len(), onnx_data.len() as u64);

    let emb = model.embed("test").expect("Failed to embed");
    assert_eq!(emb.values.len(), 1024);
}

#[test]
fn test_from_bytes_recovers_from_corrupt_file() {
    let onnx = include_bytes!("../weights/model.int4.onnx");
    let onnx_data = include_bytes!("../weights/model.int4.onnx.data");
    let tok = include_bytes!("../tokenizer/tokenizer.json");

    let dir = std::env::temp_dir().join("int4_runner_embed");
    let onnx_path = dir.join("model.int4.onnx");

    fs::create_dir_all(&dir).unwrap();
    
    // Create a "corrupt" truncated file.
    {
        let mut f = fs::File::create(&onnx_path).unwrap();
        f.write_all(b"not a valid onnx").unwrap();
    }

    // This call SHOULD succeed now because from_bytes should see the wrong size and overwrite.
    let model = EmbeddingModel::from_bytes(onnx, onnx_data, tok).expect("Failed to load and recover from corrupt file");
    
    // Verify it was overwritten
    assert_eq!(fs::metadata(&onnx_path).unwrap().len(), onnx.len() as u64);

    let emb = model.embed("test").expect("Failed to embed after recovery");
    assert_eq!(emb.values.len(), 1024);
}
