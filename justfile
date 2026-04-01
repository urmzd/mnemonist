default: check

# Initialize project
init:
    rustup component add clippy rustfmt
    sr init --merge 2>/dev/null || sr init

# Build all crates
build:
    cargo build --workspace

# Install the CLI binary
install:
    cargo install --path crates/llmem-cli

# Run all tests
test:
    cargo test --workspace

# Run clippy linter
lint:
    cargo clippy --workspace -- -D warnings

# Format all code
fmt:
    cargo fmt --all

# Check formatting without modifying
check-fmt:
    cargo fmt --all -- --check

# Record showcase with teasr
record:
    teasr showme

# Run all benchmarks
bench:
    cargo bench --workspace

# Run validation tests
validate:
    cargo test --workspace --test validation

# Quality gate: format + lint + test
check: check-fmt lint test

# Full CI gate: format + lint + build + test
ci: check-fmt lint build test
