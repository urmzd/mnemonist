# mnemonist-server

RAG HTTP server for [mnemonist](https://github.com/urmzd/mnemonist) — the open ecosystem for tool-agnostic AI agent memory.

## Install

```bash
cargo install mnemonist-server
```

## Usage

```bash
mnemonist-server  # listens on 127.0.0.1:3179
```

## Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Status, memory counts, active context |
| `GET /search?q=<query>&level=both&top_k=10` | Search memories |
| `GET /reload` | Hot-reload after context switch |

## Response format

```json
{
  "title": "Prefer Rust",
  "file": "feedback_prefer_rust.md",
  "summary": "Default to Rust for new CLI tools",
  "level": "project",
  "score": 1.0
}
```

The server is optional — mnemonist works without it. Use the server when you need HTTP-based search for RAG pipelines.

## License

Apache-2.0
