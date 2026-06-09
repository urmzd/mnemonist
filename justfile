default: check

# Download LongMemEval benchmark datasets
get-data:
    ./scripts/get_data.sh

# Initialize project
init: get-data
    rustup component add clippy rustfmt
    sr init

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
    cargo clippy --workspace --all-targets -- -D warnings

# Compile-check feature-gated targets that default features skip (mnemonist-bench).
# cuda (nvcc) and accelerate (macOS framework) are platform-bound, so not --all-features.
check-features:
    cargo check --workspace --all-targets --features mnemonist-core/bench-cli

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

# Run LongMemEval benchmark suite (all 6 experiments)
longmemeval dataset="data/longmemeval_s_cleaned.json":
    cargo run --release --bin mnemonist-bench --features bench-cli -- --dataset {{dataset}} --temporal-cycles 10

# Run specific LongMemEval experiments (e.g. just longmemeval-select 1,2)
longmemeval-select experiments dataset="data/longmemeval_s_cleaned.json":
    cargo run --release --bin mnemonist-bench --features bench-cli -- --dataset {{dataset}} --temporal-cycles 10 --experiments {{experiments}}

# Run LongMemEval benchmarks with JSON output
longmemeval-json dataset="data/longmemeval_s_cleaned.json":
    cargo run --release --bin mnemonist-bench --features bench-cli -- --dataset {{dataset}} --temporal-cycles 10 --format json

# Run validation tests
validate:
    cargo test --workspace --test validation

# Run eval suite (evals/ann/quant are features of mnemonist-core, not a separate crate)
eval:
    cargo test -p mnemonist-core --features evals,ann,quant
    cargo bench -p mnemonist-core --features evals,ann,quant

# Quality gate: format + lint + test
check: check-fmt lint check-features test

# Full CI gate: format + lint + build + test
ci: check-fmt lint check-features build test
