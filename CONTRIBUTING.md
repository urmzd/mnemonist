# Contributing

Thanks for your interest in contributing to **llmem**.

## Prerequisites

- Rust toolchain (stable)
- Git

## Getting Started

```bash
git clone https://github.com/urmzd/llmem.git
cd llmem
cargo build
cargo test
```

## Commit Convention

Conventional commits via `sr commit`:

| Prefix | Purpose |
|--------|---------|
| `feat` | New feature or memory type |
| `fix` | Bug fix or spec clarification |
| `docs` | README, spec, or guide updates |
| `refactor` | Code restructuring |
| `test` | Test additions or changes |
| `chore` | Repo maintenance |

Scope by crate: `feat(core):`, `feat(cli):`, `feat(server):`, `docs(spec):`.

## Pull Requests

1. Fork the repository
2. Create a feature branch
3. Keep PRs focused — one logical change per PR
4. For spec changes, explain the rationale in the PR description
5. Ensure `cargo test` passes
