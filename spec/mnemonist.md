# mnemonist Specification

> Version 0.2.0

## Abstract

mnemonist is a biologically-inspired memory system for AI agents. It stores memories as plain markdown files across two levels (project and global), indexes them with embeddings and approximate nearest neighbor search, and retrieves them through a multi-layer pipeline that blends semantic similarity with temporal relevance. The architecture mirrors human memory: a capacity-limited working memory (inbox), consolidation cycles that promote, associate, and decay memories, and Hebbian reinforcement that strengthens frequently-accessed memories.

## Status

Draft — seeking community feedback.

## 1. Architecture Overview

```
                        ┌─────────────────────────────────────────────────┐
                        │                   Agent / CLI                    │
                        └──────┬──────────────┬──────────────┬────────────┘
                               │              │              │
                          memorize        note/learn     remember
                               │              │              │
                   ┌───────────▼──┐    ┌──────▼──────┐  ┌───▼────────────┐
                   │  Long-term   │    │   Inbox     │  │   Retrieval    │
                   │  Memory      │◄───│  (Working   │  │   Pipeline     │
                   │  (.md files) │    │   Memory)   │  │                │
                   └──────┬───────┘    └─────────────┘  └───┬────────────┘
                          │         consolidate              │
                   ┌──────▼───────────────────┐      ┌──────▼────────────┐
                   │  Embedding Store          │      │  Two-Layer Graph  │
                   │  (.embeddings.bin)        │      │                   │
                   └──────┬───────────────────┘      │  Memory HNSW      │
                          │                           │  Code HNSW        │
                   ┌──────▼───────────────────┐      │  Inter-layer refs │
                   │  ANN Indices              │◄─────┘                   │
                   │  (.memory-index.hnsw)     │                          │
                   │  (.code-index.hnsw)       │                          │
                   └───────────────────────────┘──────────────────────────┘
```

### 1.1 Data Flow Summary

| Path | Description |
|------|-------------|
| **memorize** | Direct write to long-term memory, bypassing the inbox. Embeds immediately. Sets `strength: 1.0`. |
| **note** | Adds raw text to the inbox with `attention_score: 0.5`. No embedding. |
| **learn** | Tree-sitter code extraction → embed chunks → build code HNSW → score and push top chunks to inbox. |
| **consolidate** | Promote inbox → associate similar memories → decay stale ones → re-embed all. |
| **remember** | Semantic search (HNSW + brute-force fallback) → temporal re-ranking → ref expansion → budget-trimmed output. |

## 2. Memory Levels

| Level | Location | Scope |
|-------|----------|-------|
| Project | `~/.mnemonist/{project}/` | Per-repo corrections, decisions, context |
| Global | `~/.mnemonist/global/` | Cross-project user preferences, expertise |

- Both levels use identical directory structure and file format
- Project memory takes precedence over global on conflict
- Agents SHOULD load both `MEMORY.md` index files at session start
- The `{project}` directory name is derived from the project root's basename

## 3. Storage Layout

Each level contains:

```
MEMORY.md                  # Index — always loaded at session start
{type}_{topic}.md          # Memory files (kebab-case topic)
.embeddings.bin            # Embedding store (LMEM binary format)
.memory-index.hnsw         # Memory-layer ANN index
.code-index.hnsw           # Code-layer ANN index
.code-index.json           # Code chunk manifest (learn output)
.inbox.json                # Working memory staging area
```

### 3.1 Index File (MEMORY.md)

The index is the only file loaded automatically. It serves as a table of contents for relevance matching.

**Format:** One line per memory, as a markdown list item:

```markdown
- [Title](filename.md) — one-line summary
```

**Constraints:**

- Each line MUST be under 150 characters
- The index MUST stay under 200 lines per level (configurable via `index.max_lines`)
- Descriptions MUST be specific enough for relevance matching
- The index MUST NOT contain memory content — only pointers

### 3.2 Memory File Format

Each memory file uses YAML frontmatter followed by a markdown body:

```markdown
---
name: <kebab-case-identifier>
description: <one-line summary — used for relevance matching>
type: <user | feedback | project | reference>
created_at: <ISO 8601 timestamp>
source: <memorize | note | learn | consolidation>
strength: <f32, consolidation strength>
access_count: <u32, retrieval count>
last_accessed: <ISO 8601 timestamp>
refs:
  - <memory filename or code chunk ID>
---

<markdown content>
```

#### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Kebab-case identifier, unique within its level |
| `description` | string | One-line summary for relevance matching |
| `type` | enum | `user`, `feedback`, `project`, `reference` |

#### Cognitive Metadata

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `created_at` | string | — | ISO 8601 creation timestamp |
| `last_accessed` | string | — | ISO 8601 timestamp of last retrieval |
| `access_count` | u32 | 0 | Retrieval count (Hebbian reinforcement) |
| `strength` | f32 | 0.0 | Consolidation strength; `>= 1.0` protects from decay |
| `source` | string | — | Origin: `memorize`, `note`, `learn`, `consolidation` |
| `consolidated_from` | string[] | — | Original filenames if created via merge |
| `refs` | string[] | [] | Inter-layer edges (see [Section 7](#7-inter-layer-graph)) |

#### Body Content

- Keep files small — under 50 lines recommended
- `feedback` and `project` types SHOULD include `**Why:**` and `**How to apply:**` lines
- Convert relative dates to absolute when saving
- Use standard markdown; no custom syntax

## 4. Memory Types

### 4.1 user

Who the user is — role, expertise, preferences, communication style.

- Available at both levels
- Global: cross-project preferences
- Project: repo-specific role

### 4.2 feedback

Corrections and validated approaches. The strongest signal.

- Available at both levels
- Protected from decay at 2x the normal threshold (see [Section 8.2](#82-decay))
- Global: universal preferences
- Project: repo-specific corrections

### 4.3 project

Non-obvious project context, architectural decisions, business rationale.

- Project-level ONLY

### 4.4 reference

Pointers to external resources, tools, tracking systems.

- Available at both levels

## 5. Embedding Layer

### 5.1 Embedder Trait

```rust
pub trait Embedder: Send + Sync {
    fn embed(&self, text: &str) -> Result<Vec<f32>, Error>;
    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, Error>;
    fn dimension(&self) -> usize;
}
```

The default implementation uses `candle` with HuggingFace sentence-transformer models:

| Model | Dimensions | Notes |
|-------|-----------|-------|
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | Default |

### 5.2 Embedding Store (.embeddings.bin)

Binary format with header followed by packed entries:

```
Header:
  [4 bytes] Magic: "LMEM"
  [1 byte]  Version: u8
  [4 bytes] Dimension: u32
  [4 bytes] Count: u32

Entry (repeated):
  [2 bytes] Filename length: u16
  [N bytes] Filename: UTF-8
  [8 bytes] Content hash: u64
  [D*4 bytes] Embedding: f32 * dimension
```

- Content hashes enable incremental sync — unchanged files skip re-embedding
- Hash algorithm: `std::hash::DefaultHasher` (note: not stable across Rust versions or platforms)
- Embeddings are auto-synced on `memorize` and `consolidate`

## 6. ANN Index Layer

### 6.1 AnnIndex Trait

```rust
pub trait AnnIndex: Send + Sync {
    fn insert(&mut self, id: &str, embedding: &[f32]) -> Result<(), Error>;
    fn remove(&mut self, id: &str) -> Result<bool, Error>;
    fn search(&self, query: &[f32], top_k: usize) -> Result<Vec<SearchHit>, Error>;
    fn len(&self) -> usize;
    fn save(&self, path: &Path) -> Result<(), Error>;
}
```

### 6.2 Two-Layer Index Architecture

The system maintains two separate HNSW indices per level:

| Index | File | Built By | Contents |
|-------|------|----------|----------|
| Memory layer | `.memory-index.hnsw` | `memorize`, `consolidate` | Memory file embeddings |
| Code layer | `.code-index.hnsw` | `learn` | Source code chunk embeddings |

This separation allows memories and code to be indexed independently while enabling cross-layer retrieval through hierarchical search and ref edges (see [Section 7](#7-inter-layer-graph)).

### 6.3 HNSW (default)

Hierarchical Navigable Small World graph:

- **M**: max connections per node (default 16)
- **ef_construction**: beam width during build (default 200)
- **ef_search**: beam width during query (default 50)
- **Distance**: cosine similarity
- **Serialization**: `HNSW` magic + version + config + node data

### 6.4 IVF-Flat (alternative)

Inverted File Index with flat search within clusters:

- **n_lists**: number of k-means clusters (default 16)
- **n_probe**: clusters to search (default 10)
- **Serialization**: `~/.mnemonist/{project}/.index.ivf`

## 7. Inter-Layer Graph

The `refs` field in frontmatter stores edges between memories and between memories and code chunks, forming a sparse graph that connects the memory and code layers.

### 7.1 Edge Types

| Edge | Format | Example |
|------|--------|---------|
| Memory -> Memory | `{filename}.md` | `feedback_prefer-rust.md` |
| Memory -> Code | `{file}:{start_line}:{end_line}` | `src/lib.rs:42:58` |

### 7.2 Edge Creation

Edges are created in two ways:

1. **During consolidation (associative linking)**: A temporary HNSW is built over all memory embeddings. For each memory, the top-k neighbors (default k=10) are checked. Pairs with cosine similarity above `consolidation.merge_threshold` (default 0.85) receive bidirectional `refs` edges.

2. **During promotion from inbox**: When a `learn` item is promoted to long-term memory, its original `file_source` (file path + line range) is preserved as a code ref edge.

### 7.3 Edge Expansion During Recall

When `recall.expand_refs` is enabled (default: true), retrieving a memory also follows its ref edges:

- **Memory refs**: the referenced memory file is loaded and included in results
- **Code refs**: the referenced source lines are read from disk, truncated to `consolidation.max_memory_tokens * 4` characters

Expansion is limited to `recall.max_ref_expansions` (default 3) per memory hit.

## 8. Consolidation Pipeline

The `consolidate` command runs a sleep-like cycle with four phases:

```
  Inbox                    Long-term Memory
  ┌─────────┐   promote    ┌──────────────┐
  │ Items   │──────────────>│ New memories │
  └─────────┘              └──────┬───────┘
                                  │
                           associate (HNSW neighbor scan)
                                  │
                           ┌──────▼───────┐
                           │ Linked graph │
                           └──────┬───────┘
                                  │
                            decay (prune stale)
                                  │
                           ┌──────▼───────┐
                           │ Survivors    │
                           └──────┬───────┘
                                  │
                           re-embed + rebuild HNSW
                                  │
                           ┌──────▼───────┐
                           │ Fresh index  │
                           └──────────────┘
```

### 8.1 Phase 1: Promote

Inbox items are drained and written as long-term memory files:

| Source | Assigned Type | Strength |
|--------|--------------|----------|
| `note` | `feedback` | attention_score |
| `learn` | `reference` | attention_score |

**Body compression:** Memory bodies are cues, not copies. For code items, `extract_code_cue()` extracts only signature lines (`pub fn`, `struct`, `class`, `impl`, `trait`, `def`, `function`, `export`) up to `max_memory_tokens * 4` characters. For notes, the body is truncated. Full content is accessible via ref edges.

### 8.2 Phase 1.5: Associate

A temporary in-memory HNSW is built from all embeddings. For each memory, the top-10 neighbors are scanned. Pairs above `merge_threshold` receive bidirectional `refs` edges (deduplicated, canonically ordered).

### 8.3 Phase 2: Decay

A memory is pruned when ALL three conditions hold:

1. `days_since_last_access > decay_threshold` (feedback memories use `2 * decay_threshold`)
2. `access_count < protected_access_count` (default 5)
3. `strength < 1.0`

Memories created via `memorize` are set to `strength: 1.0` and are therefore permanently protected from decay unless manually modified.

### 8.4 Phase 3: Re-embed and Rebuild

All surviving memories are re-embedded and the `.memory-index.hnsw` is rebuilt from scratch.

## 9. Retrieval Pipeline

The `remember` command implements a multi-stage retrieval pipeline:

```
  Query
    │
    ▼
  ┌──────────────────┐
  │ Semantic Search   │  HNSW on memory + code layers
  │ (or text fallback)│  brute-force cosine if no HNSW
  └────────┬─────────┘
           │
  ┌────────▼─────────┐
  │ Hierarchical Seed │  Top memory hit's embedding
  │                   │  seeds a secondary code search
  └────────┬─────────┘
           │
  ┌────────▼─────────┐
  │ Interleave        │  Memory and code hits are
  │                   │  round-robin interleaved
  └────────┬─────────┘
           │
  ┌────────▼─────────┐
  │ Temporal Rerank   │  Blend cosine with temporal score
  └────────┬─────────┘
           │
  ┌────────▼─────────┐
  │ Ref Expansion     │  Follow refs edges to related
  │                   │  memories and code chunks
  └────────┬─────────┘
           │
  ┌────────▼─────────┐
  │ Budget Trim       │  Truncate to character budget
  │ + Hebbian Update  │  Increment access_count
  └──────────────────┘
```

### 9.1 Semantic Search

1. Embed the query using the configured embedder
2. Search `.memory-index.hnsw` (top 10) per level directory
3. Search `.code-index.hnsw` (top 10) per level directory
4. If no HNSW exists, fall back to brute-force cosine over `.embeddings.bin`
5. If semantic search returns nothing, fall back to text search (substring match on index entries)

### 9.2 Hierarchical Seeding

After the initial search, the top memory hit's embedding is used as a secondary query against the code-layer HNSW. This cross-layer seeding surfaces code chunks related to the most relevant memory, even when the original query didn't match them directly.

### 9.3 Interleaving

Memory and code hits are round-robin interleaved so that both layers contribute to the output, preventing one layer from dominating.

### 9.4 Temporal Re-ranking

Results are re-ranked using a blended score:

```
final_score = (1 - lambda) * cosine_similarity + lambda * temporal_score
```

Where `lambda` is `temporal_weight` (default 0.2) and:

```
temporal_score = recency * frequency_boost * type_weight

recency         = exp(-0.01 * days_since_last_access)
frequency_boost = min(ln(1 + access_count / age_days) / 3, 1.0)
```

**Type durability weights:**

| Type | Weight |
|------|--------|
| feedback | 1.0 |
| project | 0.8 |
| user | 0.6 |
| reference | 0.4 |

Code chunks receive no temporal scoring (cosine only).

### 9.5 Hebbian Reinforcement

Each time a memory is retrieved, its `access_count` is incremented and `last_accessed` is updated. This mirrors long-term potentiation: frequently accessed memories rank higher in temporal scoring and are protected from decay.

## 10. Working Memory (Inbox)

The inbox is a capacity-limited staging area modeled after human working memory (Miller's 7 +/- 2).

### 10.1 Structure

Stored as `.inbox.json`:

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

### 10.2 Behavior

- Items are sorted by `attention_score` descending
- When at capacity, the lowest-scored item is evicted
- `note` items receive a default attention score of 0.5
- `learn` items are scored by code heuristics (see [Section 10.3](#103-attention-scoring-for-code))
- `consolidate` drains the inbox, promoting items to long-term memory

### 10.3 Attention Scoring for Code

When `learn` ingests code via tree-sitter, chunks are scored by construct type:

| Node Kind | Score |
|-----------|-------|
| `struct_item`, `class_definition` | 0.9 |
| `impl_item`, `trait_item` | 0.85 |
| `function_item`, `function_definition` | 0.8 |
| `enum_item` | 0.75 |
| Other | 0.5 |

Public items (`pub`, `export`) receive a +0.1 bonus.

## 11. Code Indexing

The `learn` command extracts semantic code chunks using tree-sitter.

### 11.1 Supported Languages

| Language | Feature Flag |
|----------|-------------|
| Rust | `lang-rust` (default) |
| Python | `lang-python` (default) |
| JavaScript | `lang-javascript` (default) |
| TypeScript | `lang-javascript` (shared) |
| Go | `lang-go` (default) |

### 11.2 Chunking Strategy

1. Walk the project directory respecting `.gitignore` (via the `ignore` crate)
2. Parse each file with tree-sitter
3. Extract nodes matching semantic boundary kinds (functions, structs, classes, impls, traits, enums, etc.)
4. Skip nodes smaller than 3 lines
5. Split nodes larger than `max_chunk_lines` (default 100)
6. Fall back to plain text 100-line chunks for unsupported languages

### 11.3 Chunk Identity

Each chunk is identified by `{relative_file_path}:{start_line}:{end_line}`. This ID format is used in code-layer HNSW entries and in `refs` edges from memory files.

## 12. TurboQuant (Vector Quantization)

Optional vector quantization compresses embedding storage from 32-bit floats to 1-4 bits per coordinate.

### 12.1 Algorithms

**MSE Quantizer**: Normalize -> random orthogonal rotation -> Lloyd-Max scalar quantization per coordinate. Achieves distortion bound D_mse <= (sqrt(3) * pi/2) * 1/4^b.

**Prod Quantizer**: MSE at (b-1) bits + QJL 1-bit residual sign. Produces unbiased inner product estimates.

### 12.2 Compressed Store (.embeddings.lmcq)

Binary format `LMCQ` with header (magic + version + dimension + count + bit-width + algorithm) followed by packed quantized entries.

### 12.3 Configuration

| Key | Default | Description |
|-----|---------|-------------|
| `quantization.enabled` | false | Enable quantized storage |
| `quantization.bits` | 2 | Bit-width per coordinate (1-4) |
| `quantization.algorithm` | "mse" | `mse` or `prod` |

See arXiv:2504.19874 for the underlying paper.

## 13. Configuration

Configuration is layered: `~/.mnemonist/mnemonist.toml` is the global default and `./mnemonist.toml` at the project root overrides fields per-project (missing fields inherit). The CLI supports dot-notation get/set (`mnemonist config get recall.budget`), which reads and writes the global file.

### 13.1 Full Configuration Reference

```toml
[storage]
root = "~/.mnemonist"

[embedding]
provider = "candle"          # Embedding provider
model = "all-MiniLM-L6-v2"  # Model identifier

[recall]
budget = 2000                # Max output chars
priority = ["feedback", "project", "user", "reference"]
expand_refs = true           # Follow inter-layer ref edges
max_ref_expansions = 3       # Max ref expansions per memory hit

[index]
max_lines = 200              # Max entries in MEMORY.md

[code]
languages = ["rust", "python", "javascript", "go"]
max_chunk_lines = 100        # Max lines per code chunk

[quantization]
enabled = false
bits = 2                     # 1-4
algorithm = "mse"            # "mse" or "prod"
temporal_weight = 0.2        # Blend factor: 0 = pure cosine, 1 = pure temporal

[consolidation]
decay_days = 90              # Days before decay eligibility
merge_threshold = 0.85       # Cosine threshold for associative linking
protected_access_count = 5   # Min accesses to protect from decay
max_memories = 200           # Max memories per level
max_memory_tokens = 120      # Max tokens per promoted memory body

[inbox]
capacity = 7                 # Working memory size

[output]
quiet = false                # Suppress elapsed-time on stderr
```

## 15. Evaluation Harness

The `mnemonist-evals` crate provides quality metrics for embeddings, search, and quantization.

### 15.1 Search Metrics

- **Precision@k**: fraction of top-k results that are relevant
- **Recall@k**: fraction of relevant items found in top-k
- **MRR**: mean reciprocal rank of first relevant result
- **NDCG@k**: normalized discounted cumulative gain (graded relevance)

### 15.2 Embedding Quality Metrics

- **Anisotropy**: measures how uniformly embeddings occupy the space (lower is better; target < 0.3)
- **Similarity range**: spread between max and min pairwise cosine (higher is better; target > 0.3)
- **Discrimination gap**: difference between intra-class and inter-class similarity
- **Intrinsic dimensionality**: effective dimensionality via participation ratio

### 15.3 Quantization Metrics

- **Per-bit MSE**: mean squared error between original and quantized vectors
- **Cosine distortion**: cosine similarity between original and reconstructed vectors
- **Compression ratio**: storage reduction factor

### 15.4 Synthetic Datasets

Evaluation uses clustered Gaussian vectors with graded relevance judgments: grade 2 for same-cluster, grade 1 for nearest-neighbor cluster.

## 16. CLI Commands

| Command | Description |
|---------|-------------|
| `memorize <text>` | Write directly to long-term memory (strength=1.0, bypasses inbox). Memory directories auto-create on first use |
| `note <text>` | Add to inbox (attention_score=0.5) |
| `remember <query>` | Cue-based retrieval through the full pipeline |
| `learn <path>` | Tree-sitter code extraction -> embed -> score -> populate inbox |
| `consolidate` | Run the four-phase consolidation cycle (supports `--dry-run`) |
| `reflect` | List all memories with cognitive metadata and inbox summary |
| `forget <file>` | Remove a memory file, its index entry, and its embedding |
| `config <action>` | Show, get, set, init, or locate the config file |

All commands output JSON to stdout (machine-readable) and TUI to stderr (human-readable, when TTY detected or `MNEMONIST_TUI` is set).

## 17. Distance Functions

The core library provides four distance functions used across the system:

| Function | Description | Used By |
|----------|-------------|---------|
| `cosine_similarity` | Cosine similarity in [-1, 1] | HNSW, brute-force search, temporal blending |
| `dot_product` | Raw dot product | Quantization (Prod algorithm) |
| `l2_distance_squared` | Squared Euclidean distance | IVF-Flat k-means |
| `normalize` | Unit-length normalization | Quantization pre-processing |

## 18. Versioning

This specification follows semantic versioning. The current version is **0.2.0**.

## 19. License

This specification is released under the Apache License 2.0.
