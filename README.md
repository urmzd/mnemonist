<p align="center">
  <h1 align="center">llmem</h1>
  <p align="center">
    An open ecosystem for tool-agnostic AI agent memory.
    <br /><br />
    <a href="#quick-start">Quick Start</a>
    &middot;
    <a href="https://github.com/urmzd/llmem/issues">Report Bug</a>
    &middot;
    <a href="SPECIFICATION.md">Specification</a>
  </p>
</p>

<p align="center">
  <a href="https://crates.io/crates/llmem"><img src="https://img.shields.io/crates/v/llmem" alt="crates.io"></a>
</p>

## Features

- **Cognitive CLI** — commands named after memory processes: `memorize`, `remember`, `note`, `learn`, `consolidate`, `reflect`, `forget`
- **Two-level memory** — project (`~/.llmem/{project}/`) and global (`~/.llmem/global/`)
- **Working memory inbox** — capacity-limited staging area (default 7 items) with attention scoring; items promoted to long-term memory via `consolidate`
- **Memory metadata** — strength, access count, last accessed, source tracking; Hebbian reinforcement on retrieval
- **Plain markdown** with YAML frontmatter — human-readable, git-friendly
- **Typed memories** — user, feedback, project, reference
- **Semantic search** — Ollama embedder (`nomic-embed-text`) with HNSW/IVF-Flat ANN indices; auto-embeds on `memorize`
- **Code ingestion** — `learn` uses tree-sitter chunking (Rust, Python, JS/TS, Go) with attention-scored promotion to inbox
- **Consolidation** — `consolidate` promotes inbox items, decays stale memories, and re-embeds
- **TurboQuant** — optional vector quantization (1-4 bit) for compact embedding storage
- **JSON-first** — stdout for structured JSON, stderr for UX; pipe-friendly
- **Context switching** — `llmem ctx switch` swaps project memory while keeping global resident
- Works with Claude Code, Codex, Gemini, Copilot, Cursor, or any AI tool

## Install

```bash
# One-liner (auto-detects OS and arch)
curl -fsSL https://raw.githubusercontent.com/urmzd/llmem/main/install.sh | bash

# Install the RAG server instead
curl -fsSL https://raw.githubusercontent.com/urmzd/llmem/main/install.sh | bash -s -- --binary llmem-server

# Options: --tag v0.1.0, --dir ~/.local/bin, --musl (Linux)
```

Or install via Cargo:

```bash
cargo install llmem-cli          # CLI
cargo install llmem-server       # RAG server (optional)
```

Or use without tooling — just create `~/.llmem/{project}/MEMORY.md` manually.

## Quick Start

### Without tooling

```bash
mkdir -p ~/.llmem/my-project
cat > ~/.llmem/my-project/MEMORY.md << 'EOF'
- [Prefer Rust](feedback_prefer_rust.md) — default to Rust for new CLI tools
EOF

cat > ~/.llmem/my-project/feedback_prefer_rust.md << 'EOF'
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
cargo install llmem-cli
llmem init                                          # project memory
llmem init --global                                 # global memory
llmem memorize "prefer Rust for CLI tools" -t feedback
llmem note "look into async runtime choices"        # quick capture to inbox
llmem learn .                                       # ingest codebase into inbox
llmem consolidate                                   # promote inbox → long-term memory
llmem remember "rust"                               # semantic + text search
llmem reflect --all                                 # review all memories + inbox
```

## Usage

### Memory Levels

| Level | Location | Scope |
|-------|----------|-------|
| Project | `~/.llmem/{project}/` | Per-repo corrections, decisions |
| Global | `~/.llmem/global/` | Cross-project preferences, expertise |

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
| `llmem init [--global]` | Create `~/.llmem/{project}/MEMORY.md` or global |
| `llmem memorize "<point>" [-t type] [-n name]` | Deliberately encode a point into long-term memory (auto-embeds) |
| `llmem note "<point>"` | Jot a quick note into working memory inbox |
| `llmem remember "<ask>" [--budget N] [--level both]` | Recall memories by cue — semantic search with text fallback |
| `llmem learn [path] [--attend glob] [--capacity N]` | Ingest a codebase via tree-sitter; top chunks promoted to inbox |
| `llmem consolidate [--dry-run]` | Promote inbox items, decay stale memories, re-embed |
| `llmem reflect [--all] [--global]` | Introspect — review memories and inbox contents |
| `llmem forget <file>` | Deliberately forget a memory |
| `llmem ctx switch [<root>]` | Switch active project context |
| `llmem ctx show` | Show active project context |
| `llmem config init` | Create default config file |
| `llmem config show` | Show current configuration |
| `llmem config get <key>` | Get a config value (dot-notation) |
| `llmem config set <key> <value>` | Set a config value |
| `llmem config path` | Print config file path |

All commands output JSON to stdout (`{"ok": true, "data": {...}}`).

### Working Memory (Inbox)

The inbox is a capacity-limited staging area modeled after human working memory (default capacity: 7). Items enter via `note` (manual) or `learn` (code ingestion) and are scored by attention:

- Items are sorted by attention score; lowest-scored items are evicted at capacity
- `consolidate` promotes inbox items to long-term memory and clears the inbox
- Stored in `.inbox.json` alongside memory files

### Consolidation

`llmem consolidate` runs a sleep-like consolidation cycle:

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

### Configuration

Config file: `~/.llmem/config.toml` (created with `llmem config init`)

```toml
[storage]
root = "~/.llmem"

[embedding]
provider = "ollama"
host = "http://localhost:11434"
model = "nomic-embed-text"

[recall]
budget = 2000
priority = ["feedback", "project", "user", "reference"]

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

Use `llmem config set embedding.model all-minilm` to change values. Environment variables (`OLLAMA_HOST`, `OLLAMA_EMBED_MODEL`) override config.

### RAG Server

```bash
cargo install llmem-server
llmem-server  # listens on 127.0.0.1:3179
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

> Measured on Apple Silicon (M-series) with `cargo bench`. Run `cargo bench` to reproduce.
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
