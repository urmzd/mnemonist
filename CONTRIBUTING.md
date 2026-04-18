# Contributing

Thanks for your interest in contributing to **mnemonist**.

## Prerequisites

- Rust toolchain (stable)
- Git

## Getting Started

```bash
git clone https://github.com/urmzd/mnemonist.git
cd mnemonist
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

Scope by crate: `feat(core):`, `feat(cli):`, `feat(quant):`, `docs(spec):`.

## Testing

### Unit and integration tests

Run the full test suite with:

```bash
cargo test --workspace
# or
just test
```

Tests are colocated with their source in `#[cfg(test)]` modules, except for the CLI which uses a dedicated file at `crates/mnemonist-cli/tests/integration.rs`.

| Crate | Tests | What is covered |
|-------|-------|-----------------|
| `mnemonist-core` | 21 | Config TOML roundtrip and dot-notation get/set; embedding store binary format and hash-based change detection; inbox capacity eviction and JSON persistence; MEMORY.md index parsing, search, and save; memory file frontmatter parsing and markdown roundtrip; `FileBackend` store/get/remove/list |
| `mnemonist-index` | 16 | HNSW insert, remove, save/load, and recall ≥ 90% on 200 vectors; IVF-Flat insert, remove, save/load, and recall ≥ 85% on 200 vectors; cosine similarity, dot product, L2 distance, and normalization; tree-sitter Rust chunking and language-extension mapping |
| `mnemonist-quant` | 39 | Lloyd-Max codebook structure and scalar quantize/dequantize at 1–4 bits; bit-packing roundtrip for 1–4 bits including non-byte-aligned counts; `TurboQuantMse` roundtrip MSE, norm preservation, zero vector, and empirical MSE against theoretical bound; `TurboQuantProd` unbiased inner-product property and fast estimate; QJL determinism and unbiased inner-product property; rotation orthogonality, forward/inverse roundtrip, and norm preservation; compressed embedding store save/load for MSE and Prod variants |
| `mnemonist-cli` | 13 | End-to-end CLI tests that spawn the binary as a subprocess: `memorize` (with type, name, and stdin JSON), `note` (inbox capacity enforcement), `remember` (match and no-match), `reflect`, `consolidate` (dry-run and real), `forget` (success and nonexistent), `config init`/`get`/`set` |

### End-to-end validation

`scripts/validate.sh` runs every CLI command against a clean, isolated `$HOME` directory and reports pass/fail with timing for each operation. It requires release binaries built beforehand:

```bash
cargo build --release
bash scripts/validate.sh
```

The script covers all command groups in order — `config`, `memorize`, `note`, `remember`, `reflect`, `learn`, `consolidate`, `forget`, and `ctx` — and exits with the number of failures as its exit code.

## Pull Requests

1. Fork the repository
2. Create a feature branch
3. Keep PRs focused — one logical change per PR
4. For spec changes, explain the rationale in the PR description
5. Ensure `cargo test` passes
