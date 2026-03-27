"""Generate training data by running prompts against an LLM API.

Reads prompt templates from training/prompts/, sends them to the configured
LLM provider, and writes JSONL output to dataset/{code,memory}/generated/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts"
DATASET_DIR = Path(__file__).parent.parent.parent.parent / "dataset"


def load_prompt(path: Path, batch_size: int = 50) -> str:
    """Load a prompt template and substitute batch_size."""
    text = path.read_text()
    return text.replace("{batch_size}", str(batch_size))


def parse_jsonl_from_response(response: str) -> list[dict]:
    """Extract JSONL objects from an LLM response that may contain markdown fences."""
    lines = []
    in_code_block = False
    for line in response.strip().splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block or stripped.startswith("{"):
            try:
                obj = json.loads(stripped)
                lines.append(obj)
            except json.JSONDecodeError:
                continue
    return lines


def write_jsonl(data: list[dict], output_path: Path) -> int:
    """Append JSONL data to a file. Returns count written."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "a") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return len(data)


def generate_batch(
    prompt_path: Path,
    output_path: Path,
    llm_fn,
    batch_size: int = 50,
    n_batches: int = 10,
) -> int:
    """Generate data by running a prompt multiple times.

    Args:
        prompt_path: Path to the prompt template markdown file.
        output_path: Path to the output JSONL file.
        llm_fn: Callable(prompt: str) -> str that calls the LLM.
        batch_size: Number of items to request per batch.
        n_batches: Number of batches to generate.

    Returns:
        Total number of items generated.
    """
    prompt = load_prompt(prompt_path, batch_size)
    total = 0

    for i in range(n_batches):
        print(f"  Batch {i + 1}/{n_batches}...", file=sys.stderr)
        response = llm_fn(prompt)
        items = parse_jsonl_from_response(response)
        written = write_jsonl(items, output_path)
        total += written
        print(f"    → {written} items", file=sys.stderr)

    return total


# Prompt registry: maps dataset names to (prompt_file, output_file) pairs
DATASETS = {
    "code_pairs": (
        PROMPTS_DIR / "code" / "generate_code_pairs.md",
        DATASET_DIR / "code" / "generated" / "code_pairs.jsonl",
    ),
    "code_similarity": (
        PROMPTS_DIR / "code" / "generate_code_similarity.md",
        DATASET_DIR / "code" / "generated" / "code_similarity.jsonl",
    ),
    "code_search": (
        PROMPTS_DIR / "code" / "generate_code_search.md",
        DATASET_DIR / "code" / "generated" / "code_search.jsonl",
    ),
    "memory_files": (
        PROMPTS_DIR / "memory" / "generate_memory_files.md",
        DATASET_DIR / "memory" / "generated" / "memory_files.jsonl",
    ),
    "memory_queries": (
        PROMPTS_DIR / "memory" / "generate_memory_queries.md",
        DATASET_DIR / "memory" / "generated" / "memory_queries.jsonl",
    ),
    "temporal_scenarios": (
        PROMPTS_DIR / "memory" / "generate_temporal_scenarios.md",
        DATASET_DIR / "memory" / "generated" / "temporal_scenarios.jsonl",
    ),
}
