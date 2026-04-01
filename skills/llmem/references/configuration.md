# Configuration Reference

Config file: `~/.llmem/config.toml` (create with `llmem config init`).

```toml
[storage]
root = "~/.llmem"

[embedding]
provider = "fastembed"
model = "all-MiniLM-L6-v2"    # 384-dim, ~22 MB, downloads on first use

[recall]
budget = 2000                  # output character limit
priority = ["feedback", "project", "user", "reference"]
expand_refs = true             # follow inter-layer edges on recall
max_ref_expansions = 3         # how many ref hops to follow

[index]
max_lines = 200                # MEMORY.md line limit

[code]
languages = ["rust", "python", "javascript", "go"]
max_chunk_lines = 100          # max lines per code chunk

[inbox]
capacity = 7                   # working memory limit (Miller's 7+/-2)

[consolidation]
decay_days = 90                # days before a memory can be pruned
merge_threshold = 0.85         # cosine similarity for associative linking
protected_access_count = 5     # access count that shields from decay
max_memories = 200             # hard cap per level
max_memory_tokens = 120        # token limit per memory body

[quantization]
enabled = false                # TurboQuant vector compression
bits = 2                       # 1-4 bit quantization
algorithm = "mse"              # "mse" or "prod"
temporal_weight = 0.2          # recency bias in quantized search
```

All values are accessible via dot-notation:

```bash
llmem config get recall.budget          # → 2000
llmem config set recall.budget 3000     # updates config.toml
llmem config set inbox.capacity 5       # reduce working memory
```
