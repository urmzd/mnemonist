# Configuration Reference

Layered config:

- `~/.mnemonist/mnemonist.toml` — global default (create with `mnemonist config init`)
- `./mnemonist.toml` at the project root — per-project overrides; fields present here replace the global value, missing fields inherit

Legacy `~/.mnemonist/config.toml` is still read as a fallback when `mnemonist.toml` doesn't exist.

```toml
[storage]
root = "~/.mnemonist"

[embedding]
provider = "candle"            # "candle" or "none" (disables embedding; text search still works)
model = "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim, downloads from HuggingFace Hub on first use

[recall]
budget = 2000                  # output character limit (default for `recall --budget`)
expand_refs = true             # follow inter-layer edges on recall
max_ref_expansions = 3         # how many ref hops to follow
min_results = 2                # always return at least N results even past the char budget

[index]
max_lines = 200                # MEMORY.md line limit

[code]
# filename prefixes/substrings to skip (case-insensitive); trimmed here —
# `mnemonist config show` prints the full default list
exclude_patterns = ["dist", "node_modules", "target", "package-lock", ".min.js"]

[consolidation]
decay_days = 90                # days before a memory can be pruned
merge_threshold = 0.85         # cosine similarity for associative linking
protected_access_count = 5     # access count that shields from decay
max_memory_tokens = 120        # token limit per memory body
auto = true                    # spawn background `consolidate --quiet` after inbox writes
auto_stale_days = 7            # auto-consolidate when the last run is older than this

[inbox]
capacity = 10                  # working memory limit

[output]
quiet = false                  # suppress elapsed-time reporting on stderr
```

Every key listed is consumed by the CLI; keys without consumers are removed
rather than documented. A config file that fails to parse is ignored with a
stderr warning and defaults are used.

All values are accessible via dot-notation:

```bash
mnemonist config get recall.budget          # → 2000
mnemonist config set recall.budget 3000     # updates ~/.mnemonist/mnemonist.toml
mnemonist config set inbox.capacity 5       # reduce working memory
```
