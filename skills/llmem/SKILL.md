---
name: llmem
description: Persistent, tool-agnostic AI agent memory at project root and global level. Use to build and maintain evolving understanding of user preferences and project context across conversations and tools.
allowed-tools: Read Grep Glob Bash Edit Write
metadata:
  title: llmem
  category: ai
  order: 0
---

# llmem

> This skill implements the [llmem specification](../../SPECIFICATION.md).

## Purpose

Build a persistent, evolving understanding of the user by observing their notes, comments, coding style, capabilities, and questions. Memory turns one-off corrections into durable knowledge that improves every future conversation — across any AI tool.

## Storage

Two levels, both using the same format:

| Level | Location | Scope |
|-------|----------|-------|
| Project | `~/.llmem/{project}/` | Per-repo corrections, decisions, context |
| Global | `~/.llmem/global/` | Cross-project user preferences, expertise |

Project memory takes precedence over global when they conflict.

### Directory layout

```
~/.llmem/{project}/  # or ~/.llmem/global/
├── MEMORY.md            # Index — always loaded at session start
├── user_<topic>.md
├── feedback_<topic>.md
├── project_<topic>.md   # project-level only
└── reference_<topic>.md
```

### Memory file format

```markdown
---
name: <kebab-case-id>
description: <one-line — used for relevance matching>
type: <user | feedback | project | reference>
---

<content — keep concise, under 50 lines>
```

- `feedback` and `project` types include `**Why:**` and `**How to apply:**`
- Absolute dates only (never "last Thursday")

### Index format (MEMORY.md)

```markdown
- [Title](file.md) — one-line summary
```

Under 200 lines per level. Loaded at every conversation start.

### Dynamic loading

- Index files: always loaded (both levels)
- Memory files: loaded on-demand by relevance matching against descriptions
- Never load all memory files at once

## Memory Types

| Type | When | Example |
|------|------|---------|
| `user` | Expertise, preferences | "Deep Rust knowledge, new to React" |
| `feedback` | Corrections, validated approaches | "Never mock the database in tests" |
| `project` | Repo-specific context (project only) | "Auth rewrite driven by compliance" |
| `reference` | External resource pointers | "Bugs tracked in Linear project INGEST" |

## What to Learn From

### Code & Commits
- Naming conventions, architecture choices, language idioms, commit style

### Comments & Notes
- TODO/FIXME patterns, inline comments, PR descriptions

### Questions & Corrections
- Questions reveal knowledge boundaries; corrections are the strongest signal

### Capabilities & Expertise
- Languages used fluently vs areas needing scaffolding

## Rules

1. **Centralized storage** — `~/.llmem/{project}/` per project, `~/.llmem/global/` globally
2. **Observe before saving** — wait for a pattern or explicit instruction
3. **Save corrections immediately** — "don't do X" is persisted now
4. **Include why** — a rule without rationale can't handle edge cases
5. **Update, don't duplicate** — check existing memories first
6. **Verify before acting** — memory is a snapshot; current code is truth
7. **Prune stale memories** — contradictions with current state get removed

## What NOT to Save

- Code patterns visible in the codebase
- Git history (use `git log`)
- Ephemeral task state
- Content already in CLAUDE.md, AGENTS.md, etc.
- Debugging solutions

## Anti-Patterns

- Tool-specific paths (`~/.claude/`, `~/.copilot/`)
- Large memory files (split into small, focused topics)
- Storing content in the index instead of separate files
- Duplicating what the codebase already expresses

## Optional CLI

```bash
cargo install llmem-cli
llmem init              # project memory
llmem init --global     # global memory
llmem add feedback <name> -d "<description>"
llmem search "<query>"
```
