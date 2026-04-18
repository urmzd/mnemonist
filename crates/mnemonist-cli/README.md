# mnemonist-cli

CLI tool for [mnemonist](https://github.com/urmzd/mnemonist) — the open ecosystem for tool-agnostic AI agent memory.

## Install

```bash
cargo install mnemonist-cli
```

## Commands

Memory directories are created automatically on first use — `mnemonist learn .` is typically the first step.

| Command | Description |
|---------|-------------|
| `mnemonist memorize "<point>" [-t type] [-n name]` | Encode a point into long-term memory (auto-embeds) |
| `mnemonist note "<point>"` | Jot a quick note into working memory inbox |
| `mnemonist remember "<ask>" [--budget N] [--level both]` | Recall by cue — searches memory and code indices, follows refs |
| `mnemonist learn [path]` | Embed all code chunks into `.code-index.hnsw`; reports quality metrics |
| `mnemonist consolidate [--dry-run]` | Promote inbox items, decay stale memories, re-embed |
| `mnemonist reflect [--all] [--global]` | Review memories and inbox contents |
| `mnemonist forget <file>` | Forget a memory |
| `mnemonist config show\|get\|set\|init\|path` | Manage configuration |

All commands output JSON to stdout (`{"ok": true, "data": {...}}`).

## Storage

- Project memory: `~/.mnemonist/{project}/`
- Global memory: `~/.mnemonist/global/`
- Code index: `.code-index.hnsw` (project root)
- Memory index: `.memory-index.hnsw` (project root)
- Config (global): `~/.mnemonist/mnemonist.toml`
- Config (project overrides): `./mnemonist.toml` at the repo root

## License

Apache-2.0
