# mnemonist-index

ANN index implementations, code chunking, and embedding evaluation for [mnemonist](https://github.com/urmzd/mnemonist) — the open ecosystem for tool-agnostic AI agent memory.

## What's included

- **AnnIndex trait** — pluggable approximate nearest neighbor interface
- **HNSW** — Hierarchical Navigable Small World graph (default, best recall); used for `.code-index.hnsw` and `.memory-index.hnsw`
- **IVF-Flat** — Inverted File Index with flat search (faster for large datasets)
- **CodeIndex** — tree-sitter semantic chunking for Rust, Python, JS/TS, Go; plain-text fallback for unsupported file types (shell scripts, markdown, TOML, etc.)
- **Distance functions** — cosine similarity, L2 distance, dot product
- **Eval module** — embedding quality metrics: `anisotropy()`, `similarity_range()`, `discrimination_gap()`, `mean_center()`

## Usage

```rust
use mnemonist_index::{AnnIndex, hnsw::{HnswIndex, HnswConfig}};

let mut index = HnswIndex::new(384, HnswConfig::default());
index.insert("doc-1", &embedding)?;

let hits = index.search(&query_vec, 10)?;
for hit in hits {
    println!("{}: {}", hit.id, hit.score);
}
```

```rust
use mnemonist_index::eval;

let anisotropy = eval::anisotropy(&embeddings);
let range = eval::similarity_range(&embeddings);
```

## Features

Language support is feature-gated (all enabled by default):

- `lang-rust` — Rust via tree-sitter-rust
- `lang-python` — Python via tree-sitter-python
- `lang-javascript` — JavaScript/TypeScript via tree-sitter-javascript/typescript
- `lang-go` — Go via tree-sitter-go

Files without a supported tree-sitter grammar are chunked as plain text.

## License

Apache-2.0
