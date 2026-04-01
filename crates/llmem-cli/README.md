# llmem-cli

CLI tool for [llmem](https://github.com/urmzd/llmem) — the open ecosystem for tool-agnostic AI agent memory.

## Install

```bash
cargo install llmem-cli
```

## Commands

| Command | Description |
|---------|-------------|
| `llmem init [--global]` | Initialize memory directory |
| `llmem memorize "<point>" [-t type] [-n name]` | Encode a point into long-term memory (auto-embeds) |
| `llmem note "<point>"` | Jot a quick note into working memory inbox |
| `llmem remember "<ask>" [--budget N] [--level both]` | Recall by cue — searches memory and code indices, follows refs |
| `llmem learn [path]` | Embed all code chunks into `.code-index.hnsw`; reports quality metrics |
| `llmem consolidate [--dry-run]` | Promote inbox items, decay stale memories, re-embed |
| `llmem reflect [--all] [--global]` | Review memories and inbox contents |
| `llmem forget <file>` | Forget a memory |
| `llmem config show\|get\|set\|init\|path` | Manage configuration |

All commands output JSON to stdout (`{"ok": true, "data": {...}}`).

## Storage

- Project memory: `~/.llmem/{project}/`
- Global memory: `~/.llmem/global/`
- Code index: `.code-index.hnsw` (project root)
- Memory index: `.memory-index.hnsw` (project root)
- Config: `~/.llmem/config.toml`

## License

Apache-2.0
