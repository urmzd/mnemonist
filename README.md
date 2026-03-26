<p align="center">
  <h1 align="center">llmem</h1>
  <p align="center">
    An open standard for tool-agnostic AI agent memory.
    <br /><br />
    <a href="#quick-start">Quick Start</a>
    &middot;
    <a href="https://github.com/urmzd/llmem/issues">Report Bug</a>
    &middot;
    <a href="SPECIFICATION.md">Specification</a>
  </p>
</p>

## Features

- Two-level memory: project (`.llmem/`) and global (`~/.config/llmem/`)
- Plain markdown with YAML frontmatter — human-readable, git-friendly
- Dynamic loading — index always loaded, individual files read on-demand
- Typed memories: user, feedback, project, reference
- Pluggable `Embedder` trait — bring your own embedding model (ONNX, Ollama, OpenAI, custom)
- ANN index with HNSW and IVF-Flat implementations for fast semantic search
- Context switching — `llmem ctx switch` swaps project memory while keeping global resident
- No tooling required — just create files; CLI and server are optional
- Works with Claude Code, Codex, Gemini, Copilot, Cursor, or any AI tool

## Install

```bash
cargo install llmem-cli          # CLI
cargo install llmem-server       # RAG server (optional)
```

Or use without tooling — just create `.llmem/MEMORY.md` manually.

## Quick Start

### Without tooling

```bash
mkdir .llmem
cat > .llmem/MEMORY.md << 'EOF'
- [Prefer Rust](feedback_prefer_rust.md) — default to Rust for new CLI tools
EOF

cat > .llmem/feedback_prefer_rust.md << 'EOF'
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
| Project | `.llmem/` at repo root | Per-repo corrections, decisions |
| Global | `~/.config/llmem/` | Cross-project preferences, expertise |

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
| `llmem init [--global]` | Create `.llmem/MEMORY.md` or global |
| `llmem add <type> <name> -d <desc>` | Add a memory |
| `llmem list [--all]` | List memories |
| `llmem search <query>` | Search by description |
| `llmem remove <file>` | Remove a memory |
| `llmem embed [--global]` | Sync embeddings for current context |
| `llmem ctx switch [<root>]` | Switch active project context |
| `llmem ctx show` | Show active project context |

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
