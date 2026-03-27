# Code Similarity Triple Generation

Generate training triples for contrastive learning. Each triple has two code snippets and a similarity score.

## Instructions

Generate {batch_size} triples. For each triple:

1. Write code_a: a code snippet (10-30 lines)
2. Write code_b: another code snippet that is related to code_a
3. Assign a similarity score from 0.0 to 1.0

## Similarity Scale

- **1.0**: Semantically identical (same algorithm, different variable names or minor style differences)
- **0.8**: Same purpose, different implementation (e.g., iterative vs recursive)
- **0.6**: Related functionality (e.g., both handle HTTP requests but different endpoints)
- **0.4**: Same domain but different purpose (e.g., both file operations: one reads, one writes)
- **0.2**: Loosely related (e.g., both use the same library for different tasks)
- **0.0**: Completely unrelated

## Output Format (JSONL)

```json
{"language_a": "rust", "code_a": "fn sort_desc(v: &mut Vec<i32>) { v.sort_by(|a, b| b.cmp(a)); }", "language_b": "python", "code_b": "def sort_desc(lst): return sorted(lst, reverse=True)", "similarity": 0.9, "reason": "Same algorithm (descending sort) in different languages"}
```

## Distribution

- ~20% score >= 0.8 (near-identical or same algorithm)
- ~30% score 0.4-0.8 (related but different)
- ~30% score 0.1-0.4 (loosely related)
- ~20% score < 0.1 (unrelated)

Include cross-language pairs (~40% of triples should have different languages for code_a and code_b).
