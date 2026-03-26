[llmem] Open standard for tool-agnostic AI agent memory

## Project Overview

A specification and Rust implementation defining a convention for storing AI agent memory as plain markdown files. Two levels: project (`.llmem/` at repo root) and global (`~/.config/llmem/`).

## Architecture

Rust workspace with three crates:
- `llmem-core` — spec types, file parsing, index operations
- `llmem-cli` — CLI binary (`llmem`) for managing memory
- `llmem-server` — RAG HTTP server for semantic search

## Discovering Structure

Use `tree` and `ripgrep` to discover project layout. Do not rely on static listings.

## Commands

```bash
cargo build              # build all crates
cargo test               # run all tests
cargo run -p llmem-cli   # run the CLI
```

## Code Style

- Rust edition 2024
- Workspace dependencies centralized in root `Cargo.toml`
- Error handling: `thiserror` for library errors, `anyhow` for binaries
- Serialization: `serde` + `serde_yaml` for frontmatter

## Commit Guidelines

Conventional commits via `sr commit`:
- `feat(core):` / `feat(cli):` / `feat(server):` — scoped by crate
- `docs(spec):` — specification changes
- `docs(readme):` — documentation changes

## Extension Guide

To add a new memory type:
1. Add variant to `MemoryType` enum in `crates/llmem-core/src/memory.rs`
2. Update `Display` and `FromStr` implementations
3. Add the type to Section 5 of `SPECIFICATION.md`
4. Update the README memory types table
