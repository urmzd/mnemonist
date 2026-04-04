[mnemonist] Open standard for tool-agnostic AI agent memory

## Project Overview

A specification and Rust implementation defining a convention for storing AI agent memory as plain markdown files. Two levels: project (`~/.mnemonist/{project}/`) and global (`~/.mnemonist/global/`). Uses cognitive metaphors for its CLI: memorize, remember, note, learn, consolidate, reflect, forget.

## Architecture

Rust workspace with four crates:
- `mnemonist-core` — spec types, file parsing, index operations, inbox, config, embeddings
- `mnemonist-cli` — CLI binary (`mnemonist`) with cognitive commands
- `mnemonist-server` — RAG HTTP server for semantic search
- `mnemonist-index` — ANN indices (HNSW, IVF-Flat) and tree-sitter code indexing
- `mnemonist-quant` — TurboQuant vector quantization (1-4 bit)

Training pipeline in `training/` (Python): data generation, model distillation, ONNX export.

## Discovering Structure

Use `tree` and `ripgrep` to discover project layout. Do not rely on static listings.

## Commands

```bash
cargo build              # build all crates
cargo test               # run all tests
cargo run -p mnemonist-cli   # run the CLI
```

## Code Style

- Rust edition 2024
- Workspace dependencies centralized in root `Cargo.toml`
- Error handling: `thiserror` for library errors, `anyhow` for binaries
- Serialization: `serde` + `serde_yaml` for frontmatter

## Commit Guidelines

Conventional commits via `sr commit`:
- `feat(core):` / `feat(cli):` / `feat(server):` / `feat(quant):` — scoped by crate
- `docs(spec):` — specification changes
- `docs(readme):` — documentation changes

## Extension Guide

To add a new memory type:
1. Add variant to `MemoryType` enum in `crates/mnemonist-core/src/memory.rs`
2. Update `Display` and `FromStr` implementations
3. Add the type to Section 5 of `SPECIFICATION.md`
4. Update the README memory types table
