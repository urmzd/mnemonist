# Memory Retrieval Triple Generation

Generate training triples for memory search: a query, relevant memories, and irrelevant memories.

## Instructions

Generate {batch_size} triples. For each:

1. Write a natural query (the kind of thing an AI agent would search its memory for)
2. Provide 1-3 relevant memory descriptions that would answer the query
3. Provide 1-3 irrelevant memory descriptions (hard negatives)

## Query Types

- **Preference recall**: "what language does the user prefer for CLI tools?"
- **Context gathering**: "what is the current sprint goal?"
- **Guidance lookup**: "how should I handle error messages in this project?"
- **Reference finding**: "where is the API documentation?"
- **History recall**: "what was the decision on the auth middleware?"

## Output Format (JSONL)

```json
{
    "query": "what testing approach does the user prefer?",
    "relevant": [
        {"type": "feedback", "description": "Integration tests must hit a real database, not mocks", "relevance": 0.95},
        {"type": "feedback", "description": "Run tests before every commit, fail fast", "relevance": 0.7}
    ],
    "irrelevant": [
        {"type": "project", "description": "Database migration scheduled for next sprint", "reason": "Same domain (database) but about planning, not testing approach"},
        {"type": "user", "description": "User has 10 years of Python experience", "reason": "About the user but unrelated to testing preferences"}
    ]
}
```

## Hard Negative Guidelines

Irrelevant memories should be:
- Same domain or topic area (e.g., both about databases)
- Same memory type when possible
- Plausible results a naive keyword search would return
- NOT answers to the query

This trains the model to understand semantic intent, not just keyword overlap.
