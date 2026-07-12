# mnemonist Specification

> Version 0.3.0
>
> Spec v0.3.0 describes the data model; crate releases version independently.

## Abstract

mnemonist is a biologically-inspired memory system for AI agents. It stores memories as plain markdown files across two levels (project and global), indexes them with embeddings and approximate nearest neighbor search, and retrieves them through a multi-layer pipeline that blends semantic similarity with temporal relevance. The architecture mirrors human memory: a capacity-limited working memory (inbox), consolidation cycles that promote, associate, and decay memories, and Hebbian access tracking that protects frequently-used memories from decay.

## Status

Draft — seeking community feedback.

## 1. Architecture Overview

```
                        ┌─────────────────────────────────────────────────┐
                        │                   Agent / CLI                    │
                        └──────┬──────────────┬──────────────┬────────────┘
                               │              │              │
                          remember     --defer / learn     recall
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
| **remember** | Direct write to long-term memory, bypassing the inbox. Embeds immediately. Sets `strength: 1.0`. |
| **remember --defer** | Adds raw text to the inbox with `attention_score: 0.95`. No embedding until promotion. |
| **learn** | Code chunk extraction via the `ChunkingStrategy` trait → embed chunks → build code HNSW → score and push top chunks to inbox. |
| **consolidate** | Promote inbox → associate similar memories → decay stale ones → re-embed all. Runs automatically as a background job on inbox pressure or staleness (see [Section 8.5](#85-automatic-consolidation)). |
| **recall** | Semantic search (HNSW + brute-force fallback) → temporal re-ranking → ref expansion → budget-trimmed output. |

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
.consolidate.lock          # Present only while a consolidation run holds the lock
.last-consolidated         # Timestamp of the last completed consolidation
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
source: <remember | learn | consolidation>
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
| `source` | string | — | Origin: `remember`, `learn`, `consolidation` (`memorize`/`note` are legacy pre-0.3 values, still recognized) |
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
- Embeddings are auto-synced on `remember` and `consolidate`

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
| Memory layer | `.memory-index.hnsw` | `remember`, `consolidate` | Memory file embeddings |
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
| `remember` (deferred) | `feedback` | attention_score |
| `learn` | `reference` | attention_score |

**Body compression:** Memory bodies are cues, not copies. For code items, `extract_code_cue()` extracts only signature lines (`pub fn`, `struct`, `class`, `impl`, `trait`, `def`, `function`, `export`) up to `max_memory_tokens * 4` characters. For deferred items, the body is truncated. Full content is accessible via ref edges.

### 8.2 Phase 1.5: Associate

A temporary in-memory HNSW is built from all embeddings. For each memory, the top-10 neighbors are scanned. Pairs above `merge_threshold` receive bidirectional `refs` edges (deduplicated, canonically ordered).

### 8.3 Phase 2: Decay

A memory is pruned when ALL three conditions hold:

1. `days_since_last_access > decay_threshold` (feedback memories use `2 * decay_threshold`)
2. `access_count < protected_access_count` (default 5)
3. `strength < 1.0`

Memories created via direct `remember` are set to `strength: 1.0` and are therefore permanently protected from decay unless manually modified.

### 8.4 Phase 3: Re-embed and Rebuild

All surviving memories are re-embedded and the `.memory-index.hnsw` is rebuilt from scratch.

### 8.5 Automatic Consolidation

Consolidation runs as a detached background worker, modeled after `git gc --auto`. After an inbox write (`remember --defer`, `learn`), the CLI spawns `consolidate --quiet` in the background when either trigger fires:

- **Pressure**: the inbox is >= 80% full
- **Staleness**: the last completed consolidation is older than `consolidation.auto_stale_days` (default 7 days; measured via the `.last-consolidated` marker file)

Concurrency is serialized through a `.consolidate.lock` file in the memory directory. A second invocation (manual or automatic) that finds a live lock exits successfully with `{"skipped": "locked"}`; locks older than 10 minutes are treated as crashed holders and broken. The marker file `.last-consolidated` is written after each completed non-dry run.

Controls: `consolidation.auto = false` disables the trigger via config; the `MNEMONIST_NO_AUTO_CONSOLIDATE=1` environment variable disables it per-invocation (used by hermetic test suites).

## 9. Retrieval Pipeline

The `recall` command implements a multi-stage retrieval pipeline:

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
  │ Multi-signal      │  Cosine blended with freshness,
  │ Rerank            │  strength, source, connectivity
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

### 9.4 Multi-signal Re-ranking

Candidates below the calibrated similarity floor are dropped (when the floor
would eliminate everything, the top few above an absolute junk floor of 0.25
survive instead — recall that returns nothing while plausible matches exist
defeats the tool). Cosine scores are min-max normalised within the batch, then
blended with metadata:

```
final_score = sc * norm_cosine + (1 - sc) * metadata_score

metadata_score = freshness_bonus + strength_bonus + source_bonus + ref_bonus

freshness_bonus = 0.2 * exp(-0.01 * age_days)     # content age; ~half at 70d
strength_bonus  = min(strength / 2, 0.3)
source_bonus    = 0.15 (remember) | 0.10 (legacy note) | 0.05 (consolidation)
ref_bonus       = min(0.05 * ref_count, 0.15)
```

`sc` is the calibrated `semantic_confidence` from the recall profile. Code
chunks carry no metadata and receive a neutral `metadata_score` of 0.3.

**Freshness** decays on *content age* (`created_at`, rewritten on upsert), not
on access time. Its job is staleness disambiguation: when the same content
exists in several versions over time (a codebase that changes often), the
fresh version wins the near-tie. Access counts deliberately do not feed
ranking — a controlled experiment (Exp 4, pre-0.3) showed access-count
reinforcement changed zero retrieval outcomes (0/0 discordant pairs, p = 1.0);
the earlier multiplicative formula also zeroed the recency signal for any
never-accessed memory.

### 9.5 Hebbian Access Tracking

Each time a memory is retrieved, its `access_count` is incremented and
`last_accessed` is updated. These feed **decay protection only** (Section 8.3):
frequently accessed memories survive consolidation. They do not influence
ranking.

## 10. Working Memory (Inbox)

The inbox is a capacity-limited staging area for working memory (default capacity: 10).

### 10.1 Structure

Stored as `.inbox.json`:

```json
{
  "capacity": 10,
  "items": [
    {
      "id": "slugified-id",
      "content": "the observation",
      "source": "remember",
      "attention_score": 0.95,
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
- `remember --defer` items receive an attention score of 0.95 (user intent outranks auto-learned heuristics)
- `learn` items are scored by code heuristics (see [Section 10.3](#103-attention-scoring-for-code))
- `consolidate` drains the inbox, promoting items to long-term memory

### 10.3 Attention Scoring for Code

When `learn` ingests code, each chunk is scored by a simple content heuristic:

- Base score of `0.5`.
- `+0.2` if the chunk contains a public/exported item (`pub ` or `export `).
- `+0.2` for chunk length (`(end_line - start_line) / 100`, capped at `0.2`).

There is no per-construct (struct/impl/trait) scoring.

## 11. Code Indexing

The `learn` command extracts code chunks via the `ChunkingStrategy` trait. There is no tree-sitter dependency; chunking is line/paragraph based and language-agnostic, so it works uniformly across all text files.

### 11.1 Chunking Strategies

| Strategy | Behavior |
|----------|----------|
| `ParagraphChunking` (default) | Splits on blank-line boundaries (paragraphs / function gaps). Adjacent small paragraphs are merged up to `max_lines`; oversized paragraphs are split with `overlap`. |
| `FixedLineChunking` | Fixed-size sliding window of `chunk_size` lines with `overlap` between consecutive chunks. |

### 11.2 Chunking Process

1. Walk the project directory respecting `.gitignore` (via the `ignore` crate)
2. Read each text file (binary files are skipped)
3. Apply the configured `ChunkingStrategy` to split the file into chunks
4. Skip chunks smaller than `min_lines` (default 3)
5. Split chunks larger than `max_lines` (default 100), with `overlap` (default 10) between the resulting pieces

### 11.3 Chunk Identity

Each chunk is identified by `{relative_file_path}:{start_line}:{end_line}`. This ID format is used in code-layer HNSW entries and in `refs` edges from memory files.

## 12. TurboQuant (Vector Quantization)

Optional vector quantization compresses embedding storage from 32-bit floats to 1-4 bits per coordinate.

### 12.1 Algorithms

**MSE Quantizer**: Normalize -> random orthogonal rotation -> Lloyd-Max scalar quantization per coordinate. Achieves distortion bound D_mse <= (sqrt(3) * pi/2) * 1/4^b.

**Prod Quantizer**: MSE at (b-1) bits + QJL 1-bit residual sign. Produces unbiased inner product estimates.

### 12.2 Compressed Store (.embeddings.lmcq)

Binary format `LMCQ` with header (magic + version + dimension + count + bit-width + algorithm) followed by packed quantized entries.

### 12.3 Status

TurboQuant is a research/eval module: it is benchmarked (see `docs/benchmarks.md`)
but not wired into `learn`/`recall`, which store full f32 vectors. It therefore
exposes no configuration keys.

See arXiv:2504.19874 for the underlying paper.

## 13. Configuration

Configuration is layered: `~/.mnemonist/mnemonist.toml` is the global default and `./mnemonist.toml` at the project root overrides fields per-project (missing fields inherit). The CLI supports dot-notation get/set (`mnemonist config get recall.budget`), which reads and writes the global file.

### 13.1 Full Configuration Reference

```toml
[storage]
root = "~/.mnemonist"

[embedding]
provider = "candle"          # Embedding provider: "candle" or "none"
model = "sentence-transformers/all-MiniLM-L6-v2"  # HuggingFace model id

[recall]
budget = 2000                # Max output chars (default for `recall --budget`)
expand_refs = true           # Follow inter-layer ref edges
max_ref_expansions = 3       # Max ref expansions per memory hit
min_results = 2              # Always return at least N results, even past budget

[index]
max_lines = 200              # Max entries in MEMORY.md

[code]
# Filename patterns skipped during code indexing (case-insensitive).
# Trimmed here; `mnemonist config show` prints the full default list.
exclude_patterns = ["dist", "node_modules", "target", "package-lock", ".min.js"]

[consolidation]
decay_days = 90              # Days before decay eligibility
merge_threshold = 0.85       # Cosine threshold for associative linking
protected_access_count = 5   # Min accesses to protect from decay
max_memory_tokens = 120      # Max tokens per promoted memory body
auto = true                  # Auto-run consolidation in the background (git gc --auto style)
auto_stale_days = 7          # Staleness trigger for the background run

[inbox]
capacity = 10                # Working memory size

[output]
quiet = false                # Suppress elapsed-time on stderr
```

Every key above has a consumer in the implementation; keys without consumers
are removed rather than documented.

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
| `remember <text>` | Write directly to long-term memory (strength=1.0, bypasses inbox). Memory directories auto-create on first use |
| `remember --defer <text>` | Stage in the inbox (attention_score=0.95); promoted by `consolidate` |
| `recall <query>` | Cue-based retrieval through the full pipeline |
| `learn <path>` | Code chunk extraction (via `ChunkingStrategy`) -> embed -> score -> populate inbox |
| `consolidate` | Run the four-phase consolidation cycle (supports `--dry-run`); also auto-triggered in the background (see 8.5) |
| `reflect` | List all memories with cognitive metadata and inbox summary |
| `forget <file>` | Remove a memory file, its index entry, and its embedding |
| `config <action>` | Show, get, set, init, or locate the config file |

All commands output JSON to stdout (machine-readable) and TUI to stderr (human-readable, when TTY detected or `MNEMONIST_TUI` is set).

## 17. Distance Functions

The core library provides four distance functions used across the system:

| Function | Description | Used By |
|----------|-------------|---------|
| `cosine_similarity` | Cosine similarity in [-1, 1] | HNSW, brute-force search, rerank blending |
| `dot_product` | Raw dot product | Quantization (Prod algorithm) |
| `l2_distance_squared` | Squared Euclidean distance | IVF-Flat k-means |
| `normalize` | Unit-length normalization | Quantization pre-processing |

## 18. Versioning

This specification follows semantic versioning. The current version is **0.2.0**.

## 19. License

This specification is released under the Apache License 2.0.
