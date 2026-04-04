# dataset/

Training data and model artifacts for mnemonist's custom embedding models.

## Reference

- `turboquant.pdf` — [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874) (Zandieh et al., 2025). Implements the quantization backend in `crates/mnemonist-quant/`.

## Structure

```
dataset/
  turboquant.pdf              # Reference paper
  codebooks/                  # Precomputed Lloyd-Max codebook data
  code/
    generated/                # LLM-generated JSONL training data for code embeddings
    raw/                      # Optional real code samples (gitignored)
    processed/                # Deduplicated, chunked, ready for training
  memory/
    generated/                # LLM-generated JSONL training data for memory embeddings
    raw/                      # Optional real memory file samples
    processed/                # Preprocessed with temporal metadata
  models/
    code_embed_v1.onnx        # Exported code embedding model
    memory_embed_v1.onnx      # Exported memory embedding model
```

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

Both models' output embeddings are compressed with TurboQuant (implemented in `crates/mnemonist-quant/`) for efficient storage. At 384 dimensions and 2-bit quantization, each embedding is just 96 bytes (16x reduction from f32).
