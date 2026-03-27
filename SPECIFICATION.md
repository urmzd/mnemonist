# llmem Specification

> Version 0.1.0

## Abstract

llmem defines a convention for storing AI agent memory as plain markdown files in two locations: project-level (`~/.llmem/{project}/`) and global-level (`~/.llmem/global/`). It is tool-agnostic, human-readable, and git-friendly. No tooling is required — just create files.

## Status

Draft — seeking community feedback.

## 1. Memory Levels

| Level | Location | Scope |
|-------|----------|-------|
| Project | `~/.llmem/{project}/` | Per-repo corrections, decisions, context |
| Global | `~/.llmem/global/` | Cross-project user preferences, expertise |

- Both levels use the same directory structure and file format
- Project memory takes precedence over global when they conflict
- Agents SHOULD load both levels at session start
- Agents SHOULD note conflicts and ask the user whether to update the lower-priority level

## 2. Directory Structure

Each level contains:

```
MEMORY.md              # Index — always loaded at session start
user_<topic>.md        # Who the user is, expertise, preferences
feedback_<topic>.md    # Corrections and validated approaches
project_<topic>.md     # Non-obvious project context (project-level only)
reference_<topic>.md   # Pointers to external resources
```

- The directory MUST contain a `MEMORY.md` index file
- Memory files are named `{type}_{topic}.md` where topic is kebab-case
- The `~/.llmem/{project}/` directory is local to the machine
- The `~/.llmem/global/` directory is local to the machine

## 3. Index File (MEMORY.md)

The index is the only file loaded automatically. It serves as a table of contents for relevance matching.

### Format

One line per memory, as a markdown list item:

```markdown
- [Title](filename.md) — one-line summary
```

### Constraints

- Each line MUST be under 150 characters
- The index MUST stay under 200 lines per level
- Descriptions MUST be specific enough for relevance matching
- The index MUST NOT contain memory content — only pointers

### Loading

Agents MUST load both index files at conversation start:
1. `~/.llmem/global/MEMORY.md` (global)
2. `~/.llmem/{project}/MEMORY.md` (project, if present)

## 4. Memory File Format

Each memory file uses YAML frontmatter followed by a markdown body:

```markdown
---
name: <kebab-case-identifier>
description: <one-line summary — used for relevance matching>
type: <user | feedback | project | reference>
created_at: <ISO 8601 timestamp>
source: <memorize | note | learn | consolidation>
strength: <float, consolidation strength>
access_count: <int, retrieval count>
last_accessed: <ISO 8601 timestamp>
---

<markdown content>
```

### 4.1 Required Frontmatter Fields

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Short kebab-case identifier, unique within its level |
| `description` | string | One-line summary (agents use this for relevance matching) |
| `type` | enum | One of: `user`, `feedback`, `project`, `reference` |

### 4.2 Optional Cognitive Metadata

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `created_at` | string | — | ISO 8601 timestamp of creation |
| `last_accessed` | string | — | ISO 8601 timestamp of last retrieval |
| `access_count` | u32 | 0 | Number of times retrieved (Hebbian reinforcement) |
| `strength` | f32 | 0.0 | Consolidation strength — increases when memory survives consolidation |
| `source` | string | — | How the memory was created: `memorize`, `note`, `learn`, `consolidation` |
| `consolidated_from` | string[] | — | Original filenames if created via consolidation merge |

### 4.3 Body Content

- Keep files small — under 50 lines recommended
- `feedback` and `project` types SHOULD include `**Why:**` and `**How to apply:**` lines
- Convert relative dates to absolute when saving (e.g., "Thursday" → "2026-03-25")
- Use standard markdown; no custom syntax

## 5. Memory Types

### 5.1 user

Who the user is — role, expertise, preferences, communication style.

- Available at both levels
- Global: cross-project preferences ("prefers Rust, uses sr for commits")
- Project: repo-specific role ("lead on this module, new to the frontend")

### 5.2 feedback

Corrections and validated approaches. The strongest signal — save immediately when observed.

- Available at both levels
- Global: universal preferences ("never use mocks for integration tests")
- Project: repo-specific corrections ("this repo uses sqlx, not diesel")

### 5.3 project

Non-obvious project context, architectural decisions, business rationale.

- Project-level ONLY — meaningless without a specific repo
- Example: "Auth rewrite driven by compliance, not tech debt"

### 5.4 reference

Pointers to external resources, tools, tracking systems.

- Available at both levels
- Global: cross-project tools ("bugs tracked in Linear project INGEST")
- Project: repo-specific resources ("staging at staging.example.com")

## 6. Dynamic Loading

Memory files are loaded on-demand, not all at once:

1. **Always load**: both `MEMORY.md` index files at session start
2. **Load on-demand**: individual memory files when:
   - The index description matches the current task or question
   - The user explicitly references a memory ("remember when...")
   - The memory type is relevant (e.g., `feedback_*` during code review)
3. **Never load all**: avoid reading every memory file — use the index for filtering

This keeps context windows small and memory retrieval fast.

## 7. Precedence and Conflict Resolution

When memories at different levels address the same topic:

1. Project memory overrides global
2. Agents SHOULD note the conflict to the user
3. Agents MAY suggest updating the lower-priority memory
4. Users can explicitly promote a project memory to global or demote a global memory

## 8. Behavioral Rules

### 8.1 When to Save

1. **Observe before saving** — wait for a pattern or explicit instruction; don't save after one occurrence
2. **Save corrections immediately** — "don't do X" is persisted now
3. **Include rationale** — a rule without "why" cannot handle edge cases

### 8.2 When to Update

1. Check existing memories before creating new ones
2. Update, don't duplicate
3. Prune stale memories — if a memory contradicts current code or user behavior, update or remove it

### 8.3 When to Read

1. Load both index files at conversation start
2. Read relevant memory files based on index description matching
3. Verify memory against current code before acting — memory is a snapshot; current state is truth

### 8.4 What NOT to Save

- Code patterns visible in the codebase (just read the code)
- Git history (use `git log`)
- Ephemeral task state (use a task tracker)
- Content already in project root files (CLAUDE.md, AGENTS.md, etc.)
- Debugging solutions (the fix is in the code; the commit message has context)
- Personal information unrelated to the work

## 9. Anti-Patterns

- Using tool-specific memory paths (`~/.claude/`, `~/.copilot/`, etc.)
- Large memory files (split into focused, small topics)
- Over-indexing on a single instance as a permanent preference
- Storing memory content directly in the index instead of separate files
- Creating memories that duplicate what the codebase already expresses
- Never verifying stale memories against current state

## 10. Integration Guide

### 10.1 Claude Code

Add to `CLAUDE.md`:

```
Load ~/.llmem/{project}/MEMORY.md and ~/.llmem/global/MEMORY.md at conversation start.
Save learned preferences and corrections to ~/.llmem/{project}/ or ~/.llmem/global/.
```

Claude Code can read/write files directly.

### 10.2 OpenAI Codex

Reference `~/.llmem/{project}/` in `AGENTS.md` or system instructions. Codex reads project files and follows conventions when instructed.

### 10.3 Google Gemini CLI

Include a section in `AGENTS.md` pointing to `~/.llmem/{project}/`. Gemini CLI reads project files and can create/update memory.

### 10.4 GitHub Copilot

Add to `.github/copilot-instructions.md`:

```
Load ~/.llmem/{project}/MEMORY.md for project context at conversation start.
```

### 10.5 Cursor

Add to `.cursorrules`:

```
Load ~/.llmem/{project}/MEMORY.md at conversation start for project memory context.
```

### 10.6 Generic Integration

Any AI tool that reads project files can adopt this convention:

1. Read both `MEMORY.md` index files at session start
2. Use descriptions to decide which memory files to load
3. Write new memories when corrections or patterns are observed
4. Follow the precedence rules (project > global)

## 11. Embedder Trait

An `Embedder` trait enables pluggable embedding models:

```rust
pub trait Embedder: Send + Sync {
    fn embed(&self, text: &str) -> Result<Vec<f32>, Error>;
    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, Error>;
    fn dimension(&self) -> usize;
}
```

Implementations MAY use:
- Local ONNX models (e.g., all-MiniLM-L6-v2)
- Ollama embedding endpoints
- OpenAI/Google embedding APIs
- Custom trained models

## 12. Embedding Store

Embeddings are stored alongside memory files in `~/.llmem/{project}/.embeddings.bin`.

**File format**: Binary with header (magic `LMEM` + version + dimension + count) followed by packed entries (filename + content hash + float32 vector).

- Content hashes enable incremental sync — unchanged files are not re-embedded
- The file SHOULD be committed to the repository (typically <300KB)
- Both levels store their own `.embeddings.bin`

## 13. ANN Index

An `AnnIndex` trait enables pluggable approximate nearest neighbor search:

```rust
pub trait AnnIndex: Send + Sync {
    fn insert(&mut self, id: &str, embedding: &[f32]) -> Result<(), Error>;
    fn remove(&mut self, id: &str) -> Result<bool, Error>;
    fn search(&self, query: &[f32], top_k: usize) -> Result<Vec<SearchHit>, Error>;
    fn len(&self) -> usize;
    fn save(&self, path: &Path) -> Result<(), Error>;
}
```

### 13.1 HNSW (default)

Hierarchical Navigable Small World graph:
- **M**: max connections per node (default 16)
- **ef_construction**: beam width during build (default 200)
- **ef_search**: beam width during query (default 50)
- **Distance**: cosine similarity
- **Serialization**: `~/.llmem/{project}/.index.hnsw`

### 13.2 IVF-Flat (alternative)

Inverted File Index with flat search within clusters:
- **n_lists**: number of k-means clusters (default sqrt(n))
- **n_probe**: clusters to search (default 10)
- **Serialization**: `~/.llmem/{project}/.index.ivf`

## 14. Working Memory (Inbox)

The inbox is a capacity-limited staging area modeled after human working memory (Miller's 7±2). Items from `note` (manual capture) and `learn` (code ingestion) land here before consolidation promotes them to long-term memory.

### 14.1 Structure

Stored as `.inbox.json` in the memory directory:

```json
{
  "capacity": 7,
  "items": [
    {
      "id": "slugified-id",
      "content": "the observation",
      "source": "note",
      "attention_score": 0.5,
      "created_at": "2026-03-27T00:00:00Z",
      "file_source": null
    }
  ],
  "last_updated": "2026-03-27T00:00:00Z"
}
```

### 14.2 Behavior

- Items are sorted by `attention_score` descending
- When at capacity, the lowest-scored item is evicted
- `note` items receive a default attention score of 0.5
- `learn` items are scored by code heuristics (visibility, construct type)
- `consolidate` drains the inbox, promoting items to long-term memory

### 14.3 Attention Scoring for Code

When `learn` ingests code via tree-sitter, chunks are scored:

| Construct | Score |
|-----------|-------|
| `struct` / `class` | 0.9 |
| `impl` / `trait` | 0.85 |
| `function` | 0.8 |
| `enum` | 0.75 |
| Other | 0.5 |

Public items (`pub`, `export`) receive a +0.1 bonus.

## 15. Consolidation

The `consolidate` command runs a sleep-like consolidation cycle that mirrors biological memory consolidation:

### 15.1 Phases

1. **Promote** — inbox items become long-term memories with appropriate type and initial strength
2. **Decay** — memories not accessed within `consolidation.decay_days` and below `protected_access_count` are pruned; `feedback` type memories get 2x the decay threshold
3. **Re-embed** — all surviving memories are re-embedded for fresh semantic search

### 15.2 Configuration

| Key | Default | Description |
|-----|---------|-------------|
| `consolidation.decay_days` | 90 | Days since last access before decay eligible |
| `consolidation.merge_threshold` | 0.85 | Cosine similarity threshold for merging |
| `consolidation.protected_access_count` | 5 | Min accesses to protect from decay |
| `consolidation.max_memories` | 200 | Max memories per level before forced pruning |
| `inbox.capacity` | 7 | Working memory inbox size |

### 15.3 Hebbian Reinforcement

Each time a memory is retrieved via `remember`, its `access_count` is incremented and `last_accessed` is updated. Frequently accessed memories are protected from decay, mirroring long-term potentiation.

## 16. Context Switching

The `llmem ctx switch` command manages active project context:

1. Writes the project root path to `~/.llmem/.active-ctx`
2. Server reads this file to determine which project embeddings to load
3. Global embeddings stay resident; only project embeddings are swapped

### Behavior

- On context switch: project ANN index is hot-swapped, global stays resident
- Content hashes prevent unnecessary re-embedding
- Embeddings are auto-synced on `memorize` and `consolidate`

## 17. RAG Server

An optional HTTP server (`llmem-server`) provides search over memory:

- Loads ANN indices on start (both levels)
- Reads `.active-ctx` for current project context
- API:
  - `GET /health` — status, counts, active context
  - `GET /search?q=<query>&level=project|global|both&top_k=10` — search
  - `GET /reload` — hot-reload after context switch
- Response: JSON array of `{ title, file, summary, level, score }`

The convention works without the server. The server is an optional accelerator.

## 18. TurboQuant (Vector Quantization)

Optional vector quantization compresses embedding storage from 32-bit floats to 1-4 bits per coordinate:

| Key | Default | Description |
|-----|---------|-------------|
| `quantization.enabled` | false | Enable quantized storage |
| `quantization.bits` | 2 | Bit-width per coordinate (1-4) |
| `quantization.algorithm` | "mse" | Quantization algorithm: `mse` or `prod` |
| `quantization.temporal_weight` | 0.2 | Temporal re-ranking weight (0 = pure cosine, 1 = pure temporal) |

See `crates/llmem-quant/` for implementation details and arXiv:2504.19874 for the paper.

## 19. Versioning

This specification follows semantic versioning. The current version is **0.1.0**.

Changes to the specification are tracked in the repository's commit history.

## 20. License

This specification is released under the Apache License 2.0.
