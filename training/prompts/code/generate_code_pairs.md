# Code ↔ Description Pair Generation

Generate training pairs for a code embedding model. Each pair consists of a code snippet and its natural language description.

## Instructions

Generate {batch_size} code-description pairs. For each pair:

1. Write a realistic code snippet (10-50 lines) in one of: Rust, Python, JavaScript/TypeScript, Go
2. Write a concise natural language description (1-2 sentences) of what the code does
3. Vary complexity: include simple utility functions, data structures, algorithms, API handlers, error handling, async patterns
4. Vary domain: CLI tools, web servers, data processing, file I/O, testing, configuration, parsing

## Output Format (JSONL)

Each line is a JSON object:

```json
{"language": "rust", "code": "fn fibonacci(n: u32) -> u32 { ... }", "description": "Compute the nth Fibonacci number using iterative approach with O(1) space.", "difficulty": "easy"}
```

Fields:
- `language`: one of `rust`, `python`, `javascript`, `typescript`, `go`
- `code`: the complete code snippet (properly escaped)
- `description`: what the code does in plain English
- `difficulty`: `easy`, `medium`, `hard`

## Distribution

- ~25% Rust (emphasize CLI, systems, error handling patterns)
- ~25% Python (data processing, APIs, scripting)
- ~25% JavaScript/TypeScript (web, async, React patterns)
- ~25% Go (concurrency, HTTP, CLI tools)

## Quality Guidelines

- Code must be syntactically valid and idiomatic for the language
- Descriptions should be searchable — the kind of query a developer would type
- Avoid trivial examples (hello world, simple print statements)
- Include real-world patterns: error handling, configuration, logging, testing
- Each pair should be self-contained (no external dependencies beyond standard library where possible)
