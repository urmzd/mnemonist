<p align="center">
  <h1 align="center">mnemonist</h1>
  <p align="center">
    An open ecosystem for tool-agnostic AI agent memory.
    <br /><br />
    <a href="https://github.com/urmzd/mnemonist/releases">Download</a>
    &middot;
    <a href="https://github.com/urmzd/mnemonist/issues">Report Bug</a>
    &middot;
    <a href="spec/mnemonist.md">Specification</a>
  </p>
</p>

<p align="center">
  <a href="https://github.com/urmzd/mnemonist/actions/workflows/ci.yml"><img src="https://github.com/urmzd/mnemonist/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  &nbsp;
  <a href="https://crates.io/crates/mnemonist"><img src="https://img.shields.io/crates/v/mnemonist" alt="crates.io"></a>
  &nbsp;
  <a href="LICENSE"><img src="https://img.shields.io/github/license/urmzd/mnemonist" alt="License"></a>
</p>

<p align="center">
  <img src="showcase/mnemonist-demo.gif" alt="mnemonist demo" width="760">
</p>

## Contents

- [Features](#features)
- [Install](#install)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Benchmarks](#benchmarks)
- [Testing](#testing)
- [Agent Skill](#agent-skill)
- [License](#license)

## Features

- **Cognitive CLI** — commands named after memory processes: `memorize`, `remember`, `note`, `learn`, `consolidate`, `reflect`, `forget`
- **Two-level memory** — project (`~/.mnemonist/{project}/`) and global (`~/.mnemonist/global/`)
- **Working memory inbox** — capacity-limited staging area (default 7 items) with attention scoring; items promoted to long-term memory via `consolidate`
- **Memory metadata** — strength, access count, last accessed, source tracking; Hebbian reinforcement on retrieval
- **Plain markdown** with YAML frontmatter — human-readable, git-friendly
- **Typed memories** — user, feedback, project, reference
- **Local embedding** — `candle` crate with `all-MiniLM-L6-v2` (384-dim, CPU/CUDA); no external server needed; model downloads from HuggingFace Hub on first use
- **Layered graph** — three HNSW layers: code (`.code-index.hnsw`), project memory (`.memory-index.hnsw`), and global memory; inter-layer edges via `refs` frontmatter field
- **Pluggable code chunking** — `ChunkingStrategy` trait with built-in `ParagraphChunking` (blank-line boundaries) and `FixedLineChunking` (sliding window with overlap); no tree-sitter dependency
- **Cross-layer recall** — `remember` searches memory and code indices in parallel with blended relevance scoring (semantic + temporal); follows `refs` edges to surface referenced code chunks
- **Consolidation** — `consolidate` promotes inbox items, decays stale memories, and re-embeds
- **Fuzzy forget** — `forget` resolves partial and suffix matches so you don't need the full filename
- **Embedding quality metrics** — `learn` reports anisotropy and similarity_range after indexing
- **TurboQuant** — vector quantization (1-4 bit) available as a research/eval module (see the storage-footprint benchmark); not yet wired into `learn`/`remember`, which store full f32 embeddings
- **JSON-first** — stdout for structured JSON, stderr for UX; pipe-friendly
- Works with Claude Code, Codex, Gemini, Copilot, Cursor, or any AI tool

## Install

```bash
curl -fsSL https://raw.githubusercontent.com/urmzd/mnemonist/main/install.sh | sh
```

<!-- Rust developers can also install from source with `cargo install mnemonist`. -->

### Hardware acceleration

Pre-built binaries run on CPU with pure Rust matmuls — functional, but large batch operations like `learn` are slower than accelerated builds. If you have a Rust toolchain, you can build from source with hardware acceleration:

```bash
# macOS — Apple's Accelerate BLAS (~2x faster embedding throughput)
cargo install mnemonist --features accelerate

# Linux/Windows with an NVIDIA GPU
cargo install mnemonist --features cuda
```

## Quick Start

```bash
# 1. Install
curl -fsSL https://raw.githubusercontent.com/urmzd/mnemonist/main/install.sh | sh
# Or, if you have a Rust toolchain: cargo install mnemonist

# 2. Ingest the codebase — auto-creates ~/.mnemonist/{project}/ and embeds source files
mnemonist learn .

# 3. Memorize long-term knowledge
mnemonist memorize "prefer Rust for CLI tools" -t feedback
mnemonist memorize "deep Go expertise, new to React" -t user

# 4. Jot quick notes into the working memory inbox
mnemonist note "look into async runtime choices"
mnemonist note "check Linear project INGEST for pipeline bugs"

# 5. Consolidate — promote inbox to long-term memory, decay stale items, re-embed
mnemonist consolidate

# 6. Recall — semantic + text search across memories and code
mnemonist remember "rust async patterns"

# 7. Review everything
mnemonist reflect --all

# 8. Forget something you no longer need (fuzzy name matching)
mnemonist forget prefer-rust
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

### CLI at a glance

<p align="center">
  <img src="showcase/cli-help.gif" alt="mnemonist --help" width="760">
</p>

### CLI Commands

| Command | Description |
|---------|-------------|
| `mnemonist memorize "<point>" [-t type] [-n name]` | Deliberately encode a point into long-term memory (auto-embeds) |
| `mnemonist note "<point>"` | Jot a quick note into working memory inbox |
| `mnemonist remember "<ask>" [--budget N] [--level both]` | Recall memories by cue — searches memory and code indices in parallel with blended relevance scoring, follows refs |
| `mnemonist learn [path] [--attend glob] [--capacity N]` | Ingest a codebase; chunks files with `ParagraphChunking`, embeds into `.code-index.hnsw`, reports quality metrics. `--attend` only indexes files matching this glob |
| `mnemonist consolidate [--dry-run]` | Promote inbox items, decay stale memories, re-embed into `.memory-index.hnsw` |
| `mnemonist reflect [--all] [--global]` | Introspect — review memories and inbox contents |
| `mnemonist forget <file>` | Deliberately forget a memory (supports fuzzy/suffix name matching) |
| `mnemonist config init` | Create default config file |
| `mnemonist config show` | Show current configuration |
| `mnemonist config get <key>` | Get a config value (dot-notation) |
| `mnemonist config set <key> <value>` | Set a config value |
| `mnemonist config path` | Print config file path |

All commands output JSON to stdout (`{"ok": true, "data": {...}}`). stdout is always JSON; `--format json` (default) is compact, `--format human` pretty-prints it.

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

Layered config: `~/.mnemonist/mnemonist.toml` (global default, created with `mnemonist config init`) + `./mnemonist.toml` at the project root (per-project overrides; missing fields inherit).

```toml
[storage]
root = "~/.mnemonist"

[embedding]
provider = "candle"
model = "sentence-transformers/all-MiniLM-L6-v2"

[recall]
budget = 2000
expand_refs = true
max_ref_expansions = 3
min_results = 2

[index]
max_lines = 200

[code]
# Trimmed here; `mnemonist config show` prints the full default list.
exclude_patterns = ["dist", "node_modules", "target", "package-lock", ".min.js"]

[consolidation]
decay_days = 90
merge_threshold = 0.85
protected_access_count = 5
max_memory_tokens = 120

[inbox]
capacity = 7

[output]
quiet = false
```

Use `mnemonist config set recall.budget 3000` to change values. Every key shown above is read by the CLI; a test enforces that no dead keys are accepted.

See the full [Specification](spec/mnemonist.md) for details on file format, dynamic loading, precedence rules, and integration guides.

## Benchmarks

<!-- fsrc src="docs/benchmarks.md" -->
Two kinds of benchmark live here:

1. **System-level evaluation** — does the memory/RAG pipeline retrieve the right thing
   and answer correctly? Measured on real code (your repositories) and on the
   LongMemEval conversational-memory dataset.
2. **Microbenchmarks** — how fast are the individual primitives (distance kernels,
   HNSW, quantization)? Measured with `cargo bench` (criterion).

> **Provenance.** LongMemEval retrieval-side experiments: commit `b25e88e` (clean
> working tree), 2026-06-09, via `just longmemeval-json` (`mnemonist-bench --dataset
> data/longmemeval_s_cleaned.json --temporal-cycles 10 --format json`) on an
> **Apple M4 Pro** (CPU, candle embeddings, default features + `bench-cli`), embedder
> **all-MiniLM-L6-v2** (384-dim, candle + accelerate), HNSW `m=16, m0=32,
> ef_construction=200, ef_search=100`, dataset `longmemeval_s_cleaned.json`
> (19,195 sessions / 500 questions, sha256
> `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442`). The code-RAG
> results and the gpt-4o-judged QA accuracy are from the earlier 2026-05 run at `f68c13a`
> and were not rerun. The BM25 baseline: `scripts/bm25_baseline.py` (Okapi, k1=1.5,
> b=0.75, lowercase `\w+` tokens), 2026-07-08, same dataset and construct as Exp 1.
> Reproduce with the commands in each section.

### Code retrieval — RAG over real repositories

The product's headline use case: `mnemonist learn <repo>` ingests a codebase, then
`mnemonist remember "<question>"` should surface the source files that answer it. This
measures that directly with **natural-language, intent-based queries** mapped to gold
target **files**, over five real repositories spanning Rust, Go, and Python.

| repo | n | recall@1 | recall@3 | recall@5 | recall@10 | MRR | precision@5 |
|---|---|---|---|---|---|---|---|
| fsrc (Rust/Py) | 16 | 38% | 69% | 81% | 81% | 0.542 | 0.163 |
| sr (Rust) | 16 | 31% | 50% | 56% | 88% | 0.441 | 0.125 |
| teasr (Rust) | 16 | 62% | 75% | 81% | 81% | 0.703 | 0.163 |
| saige (Go) | 14 | 21% | 57% | 86% | 93% | 0.461 | 0.186 |
| mnemonist (Rust) | 16 | 19% | 38% | 56% | 62% | 0.332 | 0.125 |
| **overall (macro)** | **78** | **34%** | **58%** | **72%** | **81%** | **0.496** | **0.152** |

For ~3 in 4 natural-language queries a relevant file lands in the top 5 (recall@5 72%),
and 4 in 5 by top 10. `recall@1` (34%) is lower — the single best chunk is often a
sibling of the true answer. Retrieval is file-level strong; exact top-chunk ranking has
headroom (a code-tuned embedder or reranker would lift it). Gold sets are in
[`docs/benchmarks/rag_gold/`](/docs/benchmarks/rag_gold/) — intent-based queries with verified gold
paths. Each repo is learned into an **isolated storage root** (`HOME` override) so the
real `~/.mnemonist` is never touched.

```bash
uv run scripts/rag_eval.py \
  --gold-dir docs/benchmarks/rag_gold --repos-root ~/github \
  --out docs/benchmarks/rag_results.json --md docs/benchmarks/rag_results.md
```

### LongMemEval — conversational memory

Five experiments in `crates/mnemonist-core/src/evals/bench/` run against a LongMemEval
dataset. The number to care about is the **end-to-end QA accuracy (37.2%)** in Exp 5:
retrieval recall on this dataset is nearly saturated — a plain BM25 baseline with no
embeddings reaches 95% recall_any@5 — so the interesting signal is the gap between
finding the right session and actually answering from it.

```bash
just longmemeval                # all experiments
just longmemeval-select 2,4     # specific experiments
```

| # | Experiment | Measures |
|---|---|---|
| 1 | Vector retrieval | per-question session retrieval recall@k (NOT QA) |
| 2 | Latency scaling | index build + p50/p95/p99 query latency, 100–10k docs |
| 3 | Storage footprint | raw vs TurboQuant size + recall, 1–4 bits |
| 4 | Temporal retrieval | recall lift from Hebbian reinforcement over time |
| 5 | LongMemEval QA | real end-to-end QA accuracy (retrieve → LLM → judge) |

#### Vector retrieval recall (Exp 1) — with a BM25 reality check

Per question, an HNSW index is built from that question's ~48-session haystack and
queried. **This is retrieval recall, not QA accuracy** — and with ~48 candidates per
question it is a nearly saturated task, so the table includes a plain BM25 (Okapi,
lexical, no embeddings) baseline run on the identical construct
(`scripts/bm25_baseline.py`). Intervals are 95% Wilson (n=500); vector results are
committed in `docs/benchmarks/results.json`, BM25 in
`docs/benchmarks/bm25_baseline.json`.

| metric | vector (HNSW + MiniLM) | BM25 baseline |
|---|---|---|
| recall_all@5 | **84.8%** [81.4, 87.7] | 80.8% [77.1, 84.0] |
| recall_all@10 | **93.2%** [90.6, 95.1] | 88.2% [85.1, 90.7] |
| recall_any@5 | 96.4% [94.4, 97.7] | 95.0% [92.7, 96.6] |
| recall_any@10 | 98.2% [96.6, 99.1] | 96.8% [94.9, 98.0] |
| MRR | 0.873 | 0.879 |
| avg query | 9.7 ms (incl. embedding) | — |

The lenient recall_any@k barely separates the two systems (96.4% vs 95.0% @5,
overlapping CIs, and BM25's MRR is nominally *higher*): almost any reasonable
retriever finds *one* gold session in a 48-session haystack. Treat recall_any@5 as a
sanity check, not a headline. The vector index earns its keep on the strict
recall_all@k — every gold session in top-k — where it leads BM25 by +4.0 pp @5 and
+5.0 pp @10. And retrieval at 96% does not mean answers at 96%: the end-to-end QA
number below is the one that matters.

```bash
uv run scripts/bm25_baseline.py --dataset data/longmemeval_s_cleaned.json \
  --out docs/benchmarks/bm25_baseline.json
```

> Timing scope (corrected 2026-06-09): `embed_time_ms` (1,583,789 ms) now measures
> haystack embedding alone, and the experiment loop's wall time is reported separately
> as `total_time_ms` (1,590,953 ms); the previously committed value timed the whole
> loop. The recall numbers were reproduced exactly under the fix (96.4% / 84.8%).

#### End-to-end QA accuracy (Exp 5) — "LongMemEval for real"

Full pipeline: retrieve top-5 sessions → a reader LLM answers from the retrieved
**transcripts** → an LLM judge scores against the gold answer.

```bash
just longmemeval-select 5        # Phase A: retrieve context (--qa-output)
# Phase B+C: generate answers, then judge (needs OPENAI_API_KEY)
uv run scripts/longmemeval_qa.py all --context context.jsonl \
  --reader-model gpt-4o-mini --judge-model gpt-4o --report report.json
```

Configuration: reader `gpt-4o-mini`, judge `gpt-4o`, `top_k=5`, embedder all-MiniLM-L6-v2,
500 questions. (LongMemEval scores are configuration-dependent; a stronger reader raises
them.) Accuracy comes from the 2026-05 judged run at `f68c13a`; the LLM-judge pipeline
was not rerun for the 2026-06-09 update and is unaffected by the retrieval-side fixes.
The retrieval-recall column was reproduced exactly by the 2026-06-09 rerun.
Accuracy intervals are 95% Wilson, computed from the committed per-type counts in
`docs/benchmarks/results.json`; both the judge script and the substring scorer now emit
them on every run. The small-n rows are wide — single-session-preference at n=30 is
consistent with anything from 3.5% to 25.6%.

| question type | accuracy | 95% CI | retrieval recall | n |
|---|---|---|---|---|
| single-session-assistant | **87.5%** | [76.4, 93.8] | 96.4% | 56 |
| single-session-user | 74.3% | [63.0, 83.1] | 91.4% | 70 |
| knowledge-update | 39.7% | [29.6, 50.8] | 100% | 78 |
| multi-session | 19.5% | [13.7, 27.1] | 99.2% | 133 |
| temporal-reasoning | 18.8% | [13.1, 26.3] | 94.0% | 133 |
| single-session-preference | 10.0% | [3.5, 25.6] | 96.7% | 30 |
| **overall** | **37.2%** | **[33.1, 41.5]** | **96.4%** | **500** |

Retrieval is strong everywhere (91–100%) — the gap to QA accuracy is the *reader's*
reasoning, not the *memory's* recall. Factual single-session recall is high (74–88%);
single-session-assistant 87.5% is only possible because the QA pipeline now emits **full
transcripts** (user + assistant turns) — it previously fed user-turns only, making
assistant-sourced answers impossible. The hard categories are genuinely hard for a small
reader: temporal date-arithmetic, multi-session synthesis, and preference (the last is
largely an artifact of a factual-recall reader prompt that abstains rather than applying
the remembered preference — recall there is 96.7%).

#### Latency scaling (Exp 2)

Per-query latency over the HNSW index at increasing corpus sizes (M4 Pro, 384-dim).
Methodology: 50 warmup queries per scale point, percentiles are the median of 5 runs,
and session subsets are deterministic (sorted session IDs) so every run measures the
same documents.

| n_docs | build | p50 | p95 | p99 |
|---|---|---|---|---|
| 100 | 18 ms | 44 µs | 46 µs | 48 µs |
| 500 | 167 ms | 111 µs | 131 µs | 137 µs |
| 1,000 | 407 ms | 130 µs | 167 µs | 179 µs |
| 5,000 | 2.8 s | 148 µs | 221 µs | 253 µs |
| 10,000 | 6.0 s | 156 µs | 244 µs | **295 µs** |

Query latency stays **sub-millisecond even at 10k documents** and scales sub-linearly.

#### Temporal reinforcement (Exp 4) — performance over time

Does recall improve as memories are *used*? Over 10 consolidation cycles with Hebbian
reinforcement (access strengthens frequently-retrieved memories), global-retrieval
setting. Both arms are scored on the same held-out eval slice (n=250), and a
no-reinforcement control runs the identical retrieve-20/rerank/take-5 path with zero
access counts, so the reported delta isolates the reinforcement signal itself.

| arm | recall_any@5 |
|---|---|
| baseline (static) | 37.2% |
| control (rerank, no reinforcement) | 37.2% |
| reinforced (10 cycles) | 37.2% |
| **delta vs control** | **+0.0 pp** |

All three arms sit at 37.2% (95% Wilson [31.4, 43.3], n=250); the exact McNemar test on
the paired reinforced-vs-control hits gives p = 1.0000 with 0/0 discordant pairs — the
two arms retrieve identical top-5 sets for every eval query. The previously published
+9.4 pp compared arms over different query populations (all 500 queries vs the held-out
250) and subtracted the pure-cosine baseline, which credited the reranker's own
contribution to reinforcement; under the corrected design the reinforcement signal
contributes nothing on this dataset. (Access patterns are synthetic repeated queries,
not real logs — a controlled demonstration of the mechanism.)

#### Storage & quantization (Exp 3)

Raw vs TurboQuant (MSE) at 1–4 bits over 19,195 × 384-dim embeddings. Recall is the
**global** retrieval setting (every query vs all sessions — far harder than the
per-question haystack above, hence lower absolute numbers; the point is *degradation vs
raw*).

| storage | bytes/vector | total | compression | recall@5 | recall@10 | McNemar p vs raw (@5 / @10) | cosine distortion |
|---|---|---|---|---|---|---|---|
| raw f32 | 1536 B | 29.5 MB | 1.0× | 26.8% | 37.0% | — | 0 |
| 4-bit | 196 B | 3.76 MB | 7.8× | 27.0% | 36.8% | 1.000 / 1.000 | 0.0047 |
| 3-bit | 148 B | 2.84 MB | 10.4× | 26.2% | 35.6% | 0.581 / 0.092 | 0.0173 |
| 2-bit | 100 B | 1.92 MB | 15.4× | 26.0% | 34.6% | 0.523 / 0.017 | 0.0603 |
| 1-bit | 52 B | 1.00 MB | 29.5× | 28.8% | 35.2% | 0.121 / 0.262 | 0.2017 |

At 3–4 bits, cosine distortion is <0.02 and recall **matches the raw baseline within
noise** (4-bit recall@10 36.8% vs raw 37.0%, exact McNemar p = 1.0 at both cutoffs) —
**quantization is effectively lossless for retrieval at 8–10× compression**. The 1-bit
recall@5 cell *exceeding* raw (28.8% vs 26.8%) is sampling noise, not signal: the paired
exact McNemar test gives p = 0.121, and at n=500 the 95% Wilson intervals are raw
[23.1, 30.8] and 1-bit [25.0, 32.9], overlapping heavily. The only cell that clears
p < 0.05 is 2-bit recall@10 (34.6% vs raw 37.0%, p = 0.017, uncorrected for the eight
comparisons), consistent with a small real degradation at 2 bits. TurboQuant is
currently a research/eval module and is
**not** wired into `learn`/`remember` (which store full f32); this benchmark is the basis
for deciding whether to integrate it.

### Microbenchmarks (`cargo bench`)

Run `just bench` (all) or `cargo bench -p mnemonist-core --features evals,ann,quant`.

#### Distance Functions

| Function | 32-d | 128-d | 384-d |
|---|---|---|---|
| `cosine_similarity` | 12 ns | 59 ns | 207 ns |
| `dot_product` | 4 ns | 28 ns | 120 ns |
| `l2_distance_squared` | 5 ns | 30 ns | 125 ns |
| `normalize` | 18 ns | 82 ns | 239 ns |

#### HNSW Index (500 vectors, dim=32)

| Operation | Time |
|---|---|
| Build (500 inserts) | 32.7 ms |
| Search top-1/10/50 | 15.2 µs |
| Save / Load | 91 µs / 85 µs |

#### IVF-Flat Index (500 vectors, dim=32)

| Operation | Time |
|---|---|
| Train (k-means, 16 clusters) | 2.2 ms |
| Search top-1/10/50 | ~12 µs |
| Save / Load | 66 µs / 57 µs |

#### TurboQuant (dim=128)

| Bit-width | MSE quantize | MSE dequantize | Prod quantize | Prod dequantize |
|---|---|---|---|---|
| 1-bit | 3.9 µs | 991 ns | — | — |
| 2-bit | 3.9 µs | 988 ns | 116 µs | 141 µs |
| 3-bit | 3.9 µs | 997 ns | 115 µs | 111 µs |
| 4-bit | 4.1 µs | 998 ns | 115 µs | 111 µs |

#### Bit Packing

| Operation | 128x2b | 384x2b | 384x4b |
|---|---|---|---|
| Pack | 161 ns | 539 ns | 264 ns |
| Unpack | 90 ns | 270 ns | 241 ns |

#### Embedding Store

| Operation | 128d x 100 | 384d x 100 | 384d x 500 |
|---|---|---|---|
| `upsert` | 34.3 µs | 76.2 µs | 493.5 µs |
| `get` | 72 ns | 71 ns | 71 ns |
| `remove` | 4.0 µs | 5.5 µs | 35.7 µs |
| `save` | 46.9 µs | 61.0 µs | 177.4 µs |
| `load` | 15.4 µs | 20.9 µs | 74.1 µs |

#### Inbox

| Operation | cap=7 | cap=50 |
|---|---|---|
| `push_to_capacity` | 698 ns | 10.3 µs |
| `push_with_eviction` | 1.3 µs | 26.9 µs |
| `save` | 39.6 µs | 46.9 µs |
| `load` | 10.9 µs | 19.7 µs |
| `drain` | 688 ns | 10.4 µs |

#### Memory Index

| Operation | 10 entries | 100 entries |
|---|---|---|
| `parse_line` | 73 ns | — |
| `to_line` | 91 ns | — |
| `search` | 612 ns | 6.0 µs |
| `upsert_new` | 549 ns | 4.7 µs |
| `upsert_existing` | 508 ns | 4.5 µs |

#### Eval Functions

| Function | 32d x 50 | 128d x 50 | 384d x 20 |
|---|---|---|---|
| `anisotropy` | 15.8 µs | 73.0 µs | 40.9 µs |
| `similarity_range` | 15.8 µs | 73.0 µs | 40.9 µs |
| `mean_center` | 1.3 µs | 3.8 µs | 5.6 µs |
| `discrimination_gap` | 16.0 µs | — | — |

> Measured on Apple Silicon (M4 Pro) with `cargo bench`. Run `just bench` to reproduce.
> Raw results in [`docs/benchmarks/`](/docs/benchmarks/).
<!-- /fsrc -->

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
