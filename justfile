default: check

# Download LongMemEval benchmark datasets
get-data:
    ./scripts/get_data.sh

# Initialize project
init: get-data
    rustup component add clippy rustfmt
    sr init --merge 2>/dev/null || sr init

# Build all crates
build:
    cargo build --workspace

# Install the CLI binary
install:
    cargo install --path crates/mnemonist-cli

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

# Run LongMemEval benchmark suite (all 4 experiments)
longmemeval dataset="data/longmemeval_s_cleaned.json":
    cargo run --release --bin mnemonist-bench --features bench-cli -- --dataset {{dataset}}

# Run specific LongMemEval experiments (e.g. just longmemeval-select 1,2)
longmemeval-select experiments dataset="data/longmemeval_s_cleaned.json":
    cargo run --release --bin mnemonist-bench --features bench-cli -- --dataset {{dataset}} --experiments {{experiments}}

# Run LongMemEval benchmarks with JSON output
longmemeval-json dataset="data/longmemeval_s_cleaned.json":
    cargo run --release --bin mnemonist-bench --features bench-cli -- --dataset {{dataset}} --format json

# Run validation tests
validate:
    cargo test --workspace --test validation

# Run eval suite
eval:
    cargo test -p mnemonist-evals
    cargo bench -p mnemonist-evals

# Quality gate: format + lint + test
check: check-fmt lint test

# Full CI gate: format + lint + build + test
ci: check-fmt lint build test
