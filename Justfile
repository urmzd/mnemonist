# Build all crates
build:
    cargo build --workspace

# Install the CLI binary
install:
    cargo install --path crates/llmem-cli

# Run all tests
test:
    cargo test --workspace

# Format all code
fmt:
    cargo fmt --all
