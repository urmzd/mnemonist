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

## Features

- Two-level memory: project (`~/.llmem/{project}/`) and global (`~/.llmem/global/`)
- Plain markdown with YAML frontmatter — human-readable, git-friendly
- Dynamic loading — index always loaded, individual files read on-demand
- Typed memories: user, feedback, project, reference
- JSON-first CLI — stdout for structured JSON, stderr for UX; pipe-friendly
- Ollama embedder built-in (`nomic-embed-text`) with pluggable `Embedder` trait
- ANN index with HNSW and IVF-Flat implementations for fast semantic search
- Code indexing — tree-sitter chunking for Rust, Python, JS/TS, Go
- Hook-ready — `recall` and `learn` commands for pre/post-hook integration
- Context switching — `llmem ctx switch` swaps project memory while keeping global resident
- Works with Claude Code, Codex, Gemini, Copilot, Cursor, or any AI tool

## Install

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
llmem init                                    # project memory
llmem init --global                           # global memory
llmem add feedback prefer-rust -d "Default to Rust for new CLI tools"
llmem list --all                              # both levels
llmem search "rust"
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
| `llmem add <type> <name> -d <desc>` | Add a memory |
| `llmem learn [--stdin]` | Upsert a memory (JSON stdin or args) |
| `llmem recall --query <q>` | Retrieve relevant memories (pre-hook) |
| `llmem list [--all]` | List memories |
| `llmem search <query>` | Search by description |
| `llmem remove <file>` | Remove a memory |
| `llmem embed [--global]` | Sync embeddings via Ollama |
| `llmem code index` | Index source code with tree-sitter |
| `llmem code search <query>` | Search indexed code chunks |
| `llmem ctx switch [<root>]` | Switch active project context |
| `llmem ctx show` | Show active project context |
| `llmem config init` | Create default config file |
| `llmem config show` | Show current configuration |
| `llmem config get <key>` | Get a config value (dot-notation) |
| `llmem config set <key> <value>` | Set a config value |
| `llmem config path` | Print config file path |

All commands output JSON to stdout (`{"ok": true, "data": {...}}`).

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

## Agent Skill

This repo's conventions are available as portable agent skills in [`skills/`](skills/), following the [Agent Skills Specification](https://agentskills.io/specification).

Related standards: [AGENTS.md](https://agents.md/) · [llms.txt](https://llmstxt.org/)

## License

Apache-2.0
