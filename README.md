<p align="center">
  <h1 align="center">mnemonist</h1>
  <p align="center">
    An open ecosystem for tool-agnostic AI agent memory.
    <br /><br />
    <a href="#quick-start">Quick Start</a>
    &middot;
    <a href="https://github.com/urmzd/mnemonist/issues">Report Bug</a>
    &middot;
    <a href="SPECIFICATION.md">Specification</a>
  </p>
</p>

<p align="center">
  <a href="https://crates.io/crates/mnemonist"><img src="https://img.shields.io/crates/v/mnemonist" alt="crates.io"></a>
</p>

## Features

- **Cognitive CLI** — commands named after memory processes: `memorize`, `remember`, `note`, `learn`, `consolidate`, `reflect`, `forget`
- **Two-level memory** — project (`~/.mnemonist/{project}/`) and global (`~/.mnemonist/global/`)
- **Working memory inbox** — capacity-limited staging area (default 7 items) with attention scoring; items promoted to long-term memory via `consolidate`
- **Memory metadata** — strength, access count, last accessed, source tracking; Hebbian reinforcement on retrieval
- **Plain markdown** with YAML frontmatter — human-readable, git-friendly
- **Typed memories** — user, feedback, project, reference
- **Local embedding** — `fastembed` crate with `all-MiniLM-L6-v2` (384-dim, ~22 MB, ONNX Runtime); no external server needed; model downloads to `~/.cache/fastembed/` on first use
- **Layered graph** — three HNSW layers: code (`.code-index.hnsw`), project memory (`.memory-index.hnsw`), and global memory; inter-layer edges via `refs` frontmatter field
- **Code ingestion** — `learn` embeds all chunks into `.code-index.hnsw`; tree-sitter for Rust/Python/JS/TS/Go, plain-text fallback for shell scripts, markdown, TOML, etc.
- **Cross-layer recall** — `remember` searches memory and code indices in parallel; follows `refs` edges to surface referenced code chunks with source lines
- **Consolidation** — `consolidate` promotes inbox items, decays stale memories, and re-embeds
- **Embedding quality metrics** — `learn` reports anisotropy and similarity_range after indexing
- **TurboQuant** — optional vector quantization (1-4 bit) for compact embedding storage
- **JSON-first** — stdout for structured JSON, stderr for UX; pipe-friendly
- Works with Claude Code, Codex, Gemini, Copilot, Cursor, or any AI tool

## Install

```bash
curl -fsSL https://raw.githubusercontent.com/urmzd/mnemonist/main/install.sh | bash
```

Install the RAG server instead:

```bash
curl -fsSL https://raw.githubusercontent.com/urmzd/mnemonist/main/install.sh | bash -s -- --binary mnemonist-server
```

> Options: `--tag v0.1.0`, `--dir ~/.local/bin`, `--musl` (Linux)

Or install via Cargo:

```bash
cargo install mnemonist-cli          # CLI
cargo install mnemonist-server       # RAG server (optional)
```

Or use without tooling — just create `~/.mnemonist/{project}/MEMORY.md` manually.

## Quick Start

### Without tooling

```bash
mkdir -p ~/.mnemonist/my-project
cat > ~/.mnemonist/my-project/MEMORY.md << 'EOF'
- [Prefer Rust](feedback_prefer_rust.md) — default to Rust for new CLI tools
EOF

cat > ~/.mnemonist/my-project/feedback_prefer_rust.md << 'EOF'
---
name: prefer-rust
description: Default to Rust for new CLI tools
type: feedback
---

Use Rust for new CLI tools unless the project already uses another language.

**Why:** Fast, single binary, strong type system.

**How to apply:** When scaffolding new CLIs, start with a Cargo workspace.
EOF
```

### With the CLI

```bash
cargo install mnemonist-cli
mnemonist init                                          # project memory
mnemonist init --global                                 # global memory
mnemonist memorize "prefer Rust for CLI tools" -t feedback
mnemonist note "look into async runtime choices"        # quick capture to inbox
mnemonist learn .                                       # embed codebase into .code-index.hnsw
mnemonist consolidate                                   # promote inbox → long-term memory
mnemonist remember "rust"                               # semantic + text search
mnemonist reflect --all                                 # review all memories + inbox
```

## Usage

### Memory Levels

| Level | Location | Scope |
|-------|----------|-------|
| Project | `~/.mnemonist/{project}/` | Per-repo corrections, decisions |
| Global | `~/.mnemonist/global/` | Cross-project preferences, expertise |

Project memory takes precedence over global when they conflict.

### Memory Types

| Type | When | Example |
|------|------|---------|
| `user` | Expertise, preferences | "Deep Rust knowledge, new to React" |
| `feedback` | Corrections, validated approaches | "Never mock the database in tests" |
| `project` | Repo-specific context (project-level only) | "Auth rewrite driven by compliance" |
| `reference` | External resource pointers | "Bugs tracked in Linear project INGEST" |

### CLI Commands

| Command | Description |
|---------|-------------|
| `mnemonist init [--global]` | Create `~/.mnemonist/{project}/MEMORY.md` or global |
| `mnemonist memorize "<point>" [-t type] [-n name]` | Deliberately encode a point into long-term memory (auto-embeds) |
| `mnemonist note "<point>"` | Jot a quick note into working memory inbox |
| `mnemonist remember "<ask>" [--budget N] [--level both]` | Recall memories by cue — searches memory and code indices in parallel, follows refs |
| `mnemonist learn [path] [--attend glob] [--capacity N]` | Ingest a codebase; embeds all chunks into `.code-index.hnsw`, reports quality metrics |
| `mnemonist consolidate [--dry-run]` | Promote inbox items, decay stale memories, re-embed into `.memory-index.hnsw` |
| `mnemonist reflect [--all] [--global]` | Introspect — review memories and inbox contents |
| `mnemonist forget <file>` | Deliberately forget a memory |
| `mnemonist config init` | Create default config file |
| `mnemonist config show` | Show current configuration |
| `mnemonist config get <key>` | Get a config value (dot-notation) |
| `mnemonist config set <key> <value>` | Set a config value |
| `mnemonist config path` | Print config file path |

All commands output JSON to stdout (`{"ok": true, "data": {...}}`).

### Working Memory (Inbox)

The inbox is a capacity-limited staging area modeled after human working memory (default capacity: 7). Items enter via `note` (manual) or `learn` (code ingestion) and are scored by attention:

- Items are sorted by attention score; lowest-scored items are evicted at capacity
- `consolidate` promotes inbox items to long-term memory and clears the inbox
- Stored in `.inbox.json` alongside memory files

### Consolidation

`mnemonist consolidate` runs a sleep-like consolidation cycle:

1. **Promote** — inbox items become long-term memories with type and strength
2. **Decay** — memories not accessed within `consolidation.decay_days` (default 90) and below `protected_access_count` (default 5) are pruned
3. **Re-embed** — all surviving memories are re-embedded for fresh semantic search

Use `--dry-run` to preview what would change.

### Memory Metadata

Each memory file tracks cognitive metadata in its frontmatter:

| Field | Description |
|-------|-------------|
| `strength` | Consolidation strength (increases on survival) |
| `access_count` | Retrieval count (Hebbian reinforcement) |
| `last_accessed` | ISO 8601 timestamp of last retrieval |
| `created_at` | When the memory was first created |
| `source` | How it was created: `memorize`, `note`, `learn`, `consolidation` |
| `consolidated_from` | Original files if created via merge |
| `refs` | Inter-layer edges — code chunk IDs or memory filenames this memory links to |

### Configuration

Config file: `~/.mnemonist/config.toml` (created with `mnemonist config init`)

```toml
[storage]
root = "~/.mnemonist"

[embedding]
provider = "fastembed"
model = "all-MiniLM-L6-v2"

[recall]
budget = 2000
priority = ["feedback", "project", "user", "reference"]
expand_refs = true
max_ref_expansions = 3

[index]
max_lines = 200

[code]
languages = ["rust", "python", "javascript", "go"]
max_chunk_lines = 100

[inbox]
capacity = 7

[consolidation]
decay_days = 90
merge_threshold = 0.85
protected_access_count = 5
max_memories = 200

[quantization]
enabled = false
bits = 2
algorithm = "mse"
temporal_weight = 0.2
```

Use `mnemonist config set embedding.model all-MiniLM-L6-v2` to change values.

### RAG Server

```bash
cargo install mnemonist-server
mnemonist-server  # listens on 127.0.0.1:3179
curl "http://localhost:3179/search?q=rust&level=both"
curl "http://localhost:3179/reload"  # hot-reload after context switch
```

See the full [Specification](SPECIFICATION.md) for details on file format, dynamic loading, precedence rules, and integration guides.

## Benchmarks

<!-- embed-src src="docs/benchmarks.md" -->
### Distance Functions

| Function | 32-d | 128-d | 384-d |
|---|---|---|---|
| `cosine_similarity` | 12 ns | 59 ns | 207 ns |
| `dot_product` | 4 ns | 28 ns | 120 ns |
| `l2_distance_squared` | 5 ns | 30 ns | 125 ns |
| `normalize` | 18 ns | 82 ns | 239 ns |

### HNSW Index (500 vectors, dim=32)

| Operation | Time |
|---|---|
| Build (500 inserts) | 32.7 ms |
| Search top-1 | 15.2 µs |
| Search top-10 | 15.2 µs |
| Search top-50 | 15.2 µs |
| Save to disk | 91 µs |
| Load from disk | 85 µs |

### IVF-Flat Index (500 vectors, dim=32)

| Operation | Time |
|---|---|
| Train (k-means, 16 clusters) | 2.2 ms |
| Search top-1 | 11.9 µs |
| Search top-10 | 12.0 µs |
| Search top-50 | 12.1 µs |
| Save to disk | 66 µs |
| Load from disk | 57 µs |

### TurboQuant MSE (dim=128)

| Bit-width | Quantize | Dequantize |
|---|---|---|
| 1-bit | 3.9 µs | 991 ns |
| 2-bit | 3.9 µs | 988 ns |
| 3-bit | 3.9 µs | 997 ns |
| 4-bit | 4.1 µs | 998 ns |

### TurboQuant Prod (dim=128)

| Bit-width | Quantize | Dequantize | IP Estimate |
|---|---|---|---|
| 2-bit | 116 µs | 141 µs | 111 µs |
| 3-bit | 115 µs | 111 µs | 111 µs |
| 4-bit | 115 µs | 111 µs | 112 µs |

### Bit Packing

| Operation | 128x2b | 384x2b | 384x4b |
|---|---|---|---|
| Pack | 161 ns | 539 ns | 264 ns |
| Unpack | 90 ns | 270 ns | 241 ns |

### Embedding Store

| Operation | 128d x 100 | 384d x 100 | 384d x 500 |
|---|---|---|---|
| `upsert` | TBD | TBD | TBD |
| `get` | TBD | TBD | TBD |
| `remove` | TBD | TBD | TBD |
| `save` | TBD | TBD | TBD |
| `load` | TBD | TBD | TBD |

### Inbox

| Operation | cap=7 | cap=50 |
|---|---|---|
| `push_to_capacity` | TBD | TBD |
| `push_with_eviction` | TBD | TBD |
| `save` | TBD | TBD |
| `load` | TBD | TBD |
| `drain` | TBD | TBD |

### Memory Index

| Operation | 10 entries | 100 entries |
|---|---|---|
| `parse_line` | TBD | — |
| `to_line` | TBD | — |
| `search` | TBD | TBD |
| `upsert_new` | TBD | TBD |
| `upsert_existing` | TBD | TBD |

### Eval Functions

| Function | 32d x 50 | 128d x 50 | 384d x 20 |
|---|---|---|---|
| `anisotropy` | TBD | TBD | TBD |
| `similarity_range` | TBD | TBD | TBD |
| `mean_center` | TBD | TBD | TBD |
| `discrimination_gap` | TBD | — | — |

> Measured on Apple Silicon (M-series) with `cargo bench`. Run `just bench` to reproduce.
<!-- /embed-src -->

## Testing

```bash
just test                  # cargo test --workspace
bash scripts/validate.sh   # full E2E validation (requires release build)
```

See [CONTRIBUTING.md](CONTRIBUTING.md#testing) for what each test suite covers and per-crate test counts.

## Agent Skill

This repo's conventions are available as portable agent skills in [`skills/`](skills/), following the [Agent Skills Specification](https://agentskills.io/specification).

Related standards: [AGENTS.md](https://agents.md/) · [llms.txt](https://llmstxt.org/)

## License

Apache-2.0
