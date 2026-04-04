# Synthetic Memory File Generation

Generate realistic mnemonist memory files for training a memory-specialized embedding model.

## Instructions

Generate {batch_size} memory files. Each file should be a complete memory document with YAML frontmatter and markdown body, following the mnemonist specification.

## Memory Types

1. **user** — Information about a user's role, goals, preferences, expertise
2. **feedback** — Guidance on how to approach work (corrections and confirmations)
3. **project** — Ongoing work, goals, decisions, deadlines within a project
4. **reference** — Pointers to external resources (docs, dashboards, trackers)

## Output Format (JSONL)

```json
{"type": "feedback", "name": "prefer-rust-for-cli", "description": "Default to Rust for new CLI tools", "body": "Use Rust for new CLI tools unless the project already uses another language.\n\n**Why:** Fast, single binary, strong type system — burned by Python packaging issues on a prior project.\n\n**How to apply:** When scaffolding new CLIs, start with a Cargo workspace.", "created_at": "2026-02-15T10:30:00Z", "access_count": 12, "last_accessed_at": "2026-03-25T14:00:00Z"}
```

## Body Structure by Type

**feedback**: Rule → **Why:** (reason) → **How to apply:** (when/where)
**project**: Fact/decision → **Why:** (motivation) → **How to apply:** (how this shapes work)
**user**: Natural prose about the user's background, role, expertise
**reference**: Resource name → URL/location → when to consult it

## Diversity Requirements

- ~25% each type
- Vary domains: web dev, systems, data science, DevOps, mobile, ML, security
- Vary specificity: broad ("always use TypeScript") to narrow ("this API endpoint returns 429 after 100 req/min")
- Include temporal variation: some memories are fresh (days old), some are stale (months old)
- Vary access patterns: some frequently accessed, some rarely touched
- Body length: 2-8 lines (memories should be concise)

## Quality Guidelines

- Each memory should feel like something a real developer would save
- Descriptions must be concise enough to fit in a MEMORY.md index line (<150 chars)
- Names should be kebab-case and descriptive
- Timestamps should be realistic (working hours, reasonable date ranges)
