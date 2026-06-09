# dataset/

Training data and model artifacts for mnemonist's custom embedding models.

## Reference

- `turboquant.pdf` — [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874) (Zandieh et al., 2025). Implements the quantization backend in `crates/mnemonist-core/src/quant/`.

## Structure

- `codebooks/`: precomputed Lloyd-Max codebook data.
- `code/`: training data for the code embedding model. `generated/` holds LLM-generated JSONL, `raw/` optional real code samples (gitignored), `processed/` deduplicated and chunked data ready for training.
- `memory/`: training data for the memory embedding model. `generated/` holds LLM-generated JSONL, `raw/` optional real memory file samples, `processed/` data preprocessed with temporal metadata.
- `models/`: planned output location for the exported ONNX models (`code_embed_v1.onnx`, `memory_embed_v1.onnx`); not yet generated.

## Two Embedding Models

### 1. Code Embedding Model

Specialized for source code retrieval. Trained on:
- (code, description) pairs across Rust, Python, JS/TS, Go
- (code_A, code_B, similarity) triples for contrastive learning
- (search_query, relevant_code, irrelevant_code) triples for retrieval

### 2. Memory Embedding Model

Specialized for mnemonist memory files with temporal awareness. Trained on:
- Synthetic memory files (all 4 types: user, feedback, project, reference)
- (query, relevant_memories, irrelevant_memories) retrieval triples
- Temporal scenarios with timestamps and access patterns

## Data Generation

Training data is generated via LLM prompts (see `training/prompts/`). This gives full control over domain specificity and avoids dependency on external teacher models.

## Quantization

Both models' output embeddings are compressed with TurboQuant (implemented in `crates/mnemonist-core/src/quant/`) for efficient storage. Theoretical (not yet active in the CLI; embeddings are stored as full f32 today): at 384 dimensions and 2-bit quantization, each embedding is just 96 bytes (16x reduction from f32).
