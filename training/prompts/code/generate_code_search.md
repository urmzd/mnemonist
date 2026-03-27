# Code Search Triple Generation

Generate training triples for code retrieval: a search query, a relevant code result, and an irrelevant code result.

## Instructions

Generate {batch_size} triples. For each:

1. Write a natural language search query (how a developer would search for code)
2. Write a relevant code snippet that answers the query
3. Write an irrelevant code snippet that is plausible but does NOT answer the query (a hard negative)

## Query Types

- **How-to**: "how to read a file line by line in rust"
- **API usage**: "axum middleware for authentication"
- **Pattern**: "singleton pattern in python"
- **Bug fix**: "fix off-by-one error in binary search"
- **Concept**: "implement a thread-safe queue"
- **Comparison**: "difference between map and flat_map"

## Output Format (JSONL)

```json
{"query": "parse TOML config file in Rust", "language": "rust", "relevant_code": "use toml; ...", "irrelevant_code": "use serde_json; ...", "irrelevant_reason": "Parses JSON instead of TOML — same domain but wrong format"}
```

## Hard Negative Guidelines

Hard negatives should be:
- In the same language as the relevant code
- In a similar domain (e.g., both deal with file parsing)
- Syntactically similar but semantically different
- NOT trivially distinguishable (avoid completely unrelated code)

This trains the model to distinguish between superficially similar but semantically different code.
