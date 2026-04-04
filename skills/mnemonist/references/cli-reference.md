# CLI Reference

All commands output JSON to stdout (`{"ok": true, "data": {...}}`). Stderr carries colored UX text (only when TTY).

## Global Flags

| Flag | Description |
|------|-------------|
| `--root <path>` | Project root (default: current directory). Determines storage at `~/.mnemonist/{basename}/` |
| `--global` | Target global memory (`~/.mnemonist/global/`) instead of project |
| `--json` | Force JSON output even when not piping |

## Commands

### `mnemonist init [--global]`

Create `MEMORY.md` index at the target level. Run once per project and once globally.

### `mnemonist memorize "<point>" [-t type] [-n name] [--global] [--stdin]`

Encode a point directly into long-term memory. Auto-embeds immediately.

| Flag | Default | Description |
|------|---------|-------------|
| `-t, --memory-type` | `feedback` | One of: `user`, `feedback`, `project`, `reference` |
| `-n, --name` | auto-slugified | Kebab-case identifier for the file |
| `--stdin` | off | Read JSON input from stdin instead of positional arg |

Creates `{type}_{name}.md` and updates `MEMORY.md`, `.embeddings.bin`, and `.memory-index.hnsw`.

### `mnemonist note "<point>" [--global]`

Quick capture into the working memory inbox (`.inbox.json`). Default attention score: 0.5.

Items stay in the inbox until `consolidate` promotes them to long-term memory.

### `mnemonist remember "<ask>" [--budget N] [--level both] [--stdin]`

Recall memories by semantic cue. Searches two layers in parallel:

1. **Memory layer** — `.memory-index.hnsw` (stored memories)
2. **Code layer** — `.code-index.hnsw` (ingested source code)

| Flag | Default | Description |
|------|---------|-------------|
| `--budget` | `2000` | Output character limit |
| `--level` | `both` | `project`, `global`, or `both` |
| `--stdin` | off | Read JSON query from stdin |

Results are ranked by cosine similarity, then sorted by type priority: feedback > project > user > reference. Follows `refs` edges to surface related code chunks (up to `max_ref_expansions`).

Falls back to text search if no embeddings exist yet.

### `mnemonist learn [path] [--attend glob] [--capacity N]`

Ingest a codebase as sensory experience. Phases:

1. **Extract** — tree-sitter chunks for Rust/Python/JS/TS/Go; plain-text fallback for other files
2. **Embed** — all chunks embedded with fastembed, stored in `.code-index.hnsw`
3. **Score** — attention scoring (struct=0.9, impl/trait=0.85, function=0.8, enum=0.75; +0.1 for public)
4. **Promote** — top chunks added to inbox

| Flag | Default | Description |
|------|---------|-------------|
| `path` | `.` | Directory to ingest |
| `--attend` | none | Glob filter (e.g., `"src/**/*.rs"`) |
| `--capacity` | inbox capacity | Max items to promote to inbox |

Reports embedding quality metrics after indexing (anisotropy, similarity_range).

### `mnemonist consolidate [--dry-run] [--global]`

Run a sleep-like consolidation cycle:

1. **Promote** — inbox items become long-term memories with type and strength
2. **Link** — similar memories connected via `refs` (cosine similarity >= `merge_threshold`)
3. **Decay** — prune memories not accessed within `decay_days` (default 90), with `access_count < protected_access_count` (default 5), and `strength < 1.0`. Feedback type gets 2x threshold (180 days)
4. **Re-embed** — rebuild `.memory-index.hnsw` from all surviving memories

Use `--dry-run` to preview changes without applying them.

### `mnemonist reflect [--all] [--global]`

Review all memories and inbox contents at the target level(s).

| Flag | Description |
|------|-------------|
| `--all` | Show both project and global |
| `--global` | Show global only |

### `mnemonist forget <file> [--global]`

Remove a memory file and its embedding. Argument is the filename (e.g., `feedback_old_approach.md`).

### `mnemonist config <action>`

| Subcommand | Description |
|------------|-------------|
| `init` | Create default `~/.mnemonist/config.toml` |
| `show` | Display full config as TOML |
| `get <key>` | Get value by dot-notation (e.g., `embedding.model`) |
| `set <key> <value>` | Set value with type preservation |
| `path` | Print config file path |
