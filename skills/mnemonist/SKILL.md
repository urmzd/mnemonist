---
name: mnemonist
description: >
  Persistent, tool-agnostic AI agent memory using the mnemonist CLI and file
  format. Use when managing memories, recalling context, ingesting code,
  running consolidation, or configuring mnemonist — even if the user just says
  "remember this", "save that", "what do you know about", or "forget".
allowed-tools: Read Grep Glob Bash Edit Write
metadata:
  title: mnemonist
  category: ai
  order: 0
---

# mnemonist

> This skill implements the [mnemonist specification](../../SPECIFICATION.md).

## Purpose

Build persistent, evolving understanding of the user and their projects by storing memories as plain markdown files with YAML frontmatter. Memory turns one-off corrections into durable knowledge that improves every future conversation — across any AI tool.

## Storage Model

Two levels, same format:

| Level | Location | Scope |
|-------|----------|-------|
| Project | `~/.mnemonist/{project}/` | Per-repo corrections, decisions, context |
| Global | `~/.mnemonist/global/` | Cross-project preferences, expertise |

Project memory takes precedence over global when they conflict.

```
~/.mnemonist/{project}/
├── MEMORY.md              # Index — always loaded at session start
├── {type}_{name}.md       # Memory files
├── .inbox.json            # Working memory inbox
├── .embeddings.bin        # Embedding vectors
├── .memory-index.hnsw     # Semantic index over memories
└── .code-index.hnsw       # Semantic index over ingested code
```

## Memory Types

| Type | When | Example |
|------|------|---------|
| `user` | Expertise, preferences | "Deep Rust knowledge, new to React" |
| `feedback` | Corrections, validated approaches | "Never mock the database in tests" |
| `project` | Repo-specific context (project only) | "Auth rewrite driven by compliance" |
| `reference` | External resource pointers | "Bugs tracked in Linear project INGEST" |

Priority order for search results: feedback > project > user > reference.

## Memory File Format

```markdown
---
name: <kebab-case-id>
description: <one-line — used for relevance matching>
type: <user | feedback | project | reference>
---

<content — concise, under 50 lines>
```

- `feedback` and `project` types should include `**Why:**` and `**How to apply:**` lines
- Always use absolute dates (never "last Thursday")
- Filename pattern: `{type}_{name}.md`

## Index Format (MEMORY.md)

```markdown
- [Title](file.md) — one-line summary
```

Under 200 lines per level. Loaded at every conversation start. Never write memory content directly into the index.

## Core Workflows

### 1. Setup

Memory directories are created automatically on first use — the first step is usually `mnemonist learn .` to ingest the codebase.

```bash
mnemonist config init             # optional: create default ~/.mnemonist/mnemonist.toml
```

### 2. Capturing Memories

**Deliberate encoding** — goes straight to long-term memory:
```bash
mnemonist memorize "prefer Rust for CLI tools" -t feedback
mnemonist memorize "user is a senior backend engineer" -t user -n senior-backend
mnemonist memorize "deploy freeze until 2026-04-05" -t project --global
```

**Quick capture** — goes to working memory inbox:
```bash
mnemonist note "look into async runtime choices"
mnemonist note "might need to refactor auth module"
```

### 3. Ingesting Code

```bash
mnemonist learn .                           # ingest entire project
mnemonist learn src/ --attend "**/*.rs"     # only Rust files in src/
```

This extracts code chunks via tree-sitter (Rust, Python, JS/TS, Go) with plain-text fallback, embeds them into `.code-index.hnsw`, and promotes high-attention chunks to the inbox.

### 4. Recalling

```bash
mnemonist remember "authentication"               # search both levels
mnemonist remember "rust patterns" --level project # project only
mnemonist remember "user preferences" --budget 500 # compact output
```

Searches memory and code indices in parallel. Follows `refs` edges to surface related code chunks. Falls back to text search if no embeddings exist.

### 5. Consolidation

```bash
mnemonist consolidate --dry-run   # preview what would change
mnemonist consolidate             # run the full cycle
```

Consolidation is a sleep-like cycle: promote inbox items to long-term memory, link similar memories via `refs`, decay stale memories, and rebuild embeddings.

### 6. Review and Cleanup

```bash
mnemonist reflect                 # project memories + inbox
mnemonist reflect --all           # both levels
mnemonist forget feedback_old.md  # remove a specific memory
```

## JSON Output

All commands write structured JSON to stdout:

```json
{"ok": true, "data": { ... }}
{"ok": false, "error": "message"}
```

Stderr carries colored UX text (only when TTY). This makes mnemonist pipe-friendly for agent integrations.

## Working Memory (Inbox)

The inbox is a capacity-limited staging area (default: 7 items) modeled after human working memory:

- Items enter via `note` (manual, attention=0.5) or `learn` (code ingestion, scored by construct type)
- Lowest-scored items are evicted when capacity is reached
- `consolidate` promotes inbox items to long-term memory and clears the inbox
- Stored in `.inbox.json`

## Two-Layer Semantic Search

`remember` searches two HNSW indices in parallel:

1. **Memory layer** — `.memory-index.hnsw` (all stored memories)
2. **Code layer** — `.code-index.hnsw` (ingested source code chunks)

Results are interleaved by relevance, then sorted by type priority. The `refs` frontmatter field stores inter-layer edges — when a memory references a code chunk or another memory, `remember` follows those edges (up to `max_ref_expansions` hops) to surface related content.

## Rules for Saving Memories

1. **Observe before saving** — wait for a pattern or explicit instruction
2. **Save corrections immediately** — "don't do X" is persisted now
3. **Include why** — a rule without rationale can't handle edge cases
4. **Update, don't duplicate** — check existing memories first via `reflect` or `remember`
5. **Verify before acting on memory** — memory is a snapshot; current code is truth
6. **Prune stale memories** — contradictions with current state get removed via `forget`

## What NOT to Save

- Code patterns visible in the codebase (read the code instead)
- Git history (use `git log`)
- Ephemeral task state (use tasks or plans)
- Content already in CLAUDE.md, AGENTS.md, etc.
- Debugging solutions (the fix is in the code)

## Gotchas

- First `memorize` or `learn` downloads the embedding model from HuggingFace Hub
- `--root` determines which project directory under `~/.mnemonist/` is used — defaults to cwd basename
- `memorize` auto-embeds; `note` does not (items embed during `consolidate`)
- `learn` overwrites the code index entirely on each run — it's a full re-ingest
- Feedback memories get 2x decay protection (180 days vs 90) during consolidation
- `remember` falls back to text search when no `.memory-index.hnsw` exists yet

## Reference

For detailed CLI flags and options, read `references/cli-reference.md`.
For full configuration options, read `references/configuration.md`.
