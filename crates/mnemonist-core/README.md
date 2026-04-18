# mnemonist-core

Core types and operations for [mnemonist](https://github.com/urmzd/mnemonist) — the open ecosystem for tool-agnostic AI agent memory.

## What's included

- **Memory types** — `MemoryFile`, `Frontmatter`, `MemoryType` (user, feedback, project, reference)
- **Index** — `MemoryIndex` for managing `MEMORY.md` index files with add/upsert/remove/search
- **Embedder trait** — pluggable embedding interface with built-in `CandleEmbedder` (Metal/CUDA/CPU)
- **Embedding store** — binary format for persisting embeddings alongside memory files
- **File backend** — `FileBackend` implementing `MemoryBackend` for file-based storage
- **Config** — `Config` struct for `~/.mnemonist/mnemonist.toml` management (layered with project-root `mnemonist.toml`)

## Usage

```rust
use mnemonist_core::{Config, MemoryIndex, MemoryFile, CandleEmbedder, Embedder};

// Load config and resolve paths
let config = Config::load();
let project_dir = config.project_dir(std::path::Path::new("."));

// Load memory index
let index = MemoryIndex::load(&project_dir)?;
let results = index.search("rust");

// Embed locally with candle (Metal/CUDA/CPU, no server needed)
let embedder = CandleEmbedder::default_model()?;
let vector = embedder.embed("some text")?;
```

## License

Apache-2.0
