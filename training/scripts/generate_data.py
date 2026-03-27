#!/usr/bin/env python3
"""Orchestrate training data generation via LLM prompts.

Usage:
    python scripts/generate_data.py --provider anthropic --batches 10
    python scripts/generate_data.py --provider ollama --model llama3 --batches 5
    python scripts/generate_data.py --datasets code_pairs,memory_files --batches 20
"""

from __future__ import annotations

import argparse
import os
import sys

# Add training/ to path
sys.path.insert(0, str(__file__ and os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.datagen.generate import DATASETS, generate_batch


def make_anthropic_fn(model: str = "claude-sonnet-4-20250514"):
    """Create an LLM function using Anthropic API."""
    import anthropic

    client = anthropic.Anthropic()

    def call(prompt: str) -> str:
        message = client.messages.create(
            model=model,
            max_tokens=8192,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text

    return call


def make_ollama_fn(model: str = "llama3"):
    """Create an LLM function using local Ollama."""
    import json
    import urllib.request

    def call(prompt: str) -> str:
        data = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode()
        req = urllib.request.Request(
            "http://localhost:11434/api/generate",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read())
        return result.get("response", "")

    return call


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate training data via LLM prompts")
    parser.add_argument(
        "--provider", choices=["anthropic", "ollama"], default="anthropic",
        help="LLM provider",
    )
    parser.add_argument("--model", default=None, help="Model name override")
    parser.add_argument("--batches", type=int, default=10, help="Number of batches per dataset")
    parser.add_argument("--batch-size", type=int, default=50, help="Items per batch")
    parser.add_argument(
        "--datasets", default=None,
        help="Comma-separated dataset names (default: all)",
    )
    args = parser.parse_args()

    if args.provider == "anthropic":
        model = args.model or "claude-sonnet-4-20250514"
        llm_fn = make_anthropic_fn(model)
    elif args.provider == "ollama":
        model = args.model or "llama3"
        llm_fn = make_ollama_fn(model)

    dataset_names = args.datasets.split(",") if args.datasets else list(DATASETS.keys())

    for name in dataset_names:
        if name not in DATASETS:
            print(f"Unknown dataset: {name}. Available: {list(DATASETS.keys())}", file=sys.stderr)
            continue

        prompt_path, output_path = DATASETS[name]
        print(f"\n{'=' * 60}", file=sys.stderr)
        print(f"Generating: {name}", file=sys.stderr)
        print(f"  Prompt: {prompt_path}", file=sys.stderr)
        print(f"  Output: {output_path}", file=sys.stderr)

        total = generate_batch(
            prompt_path, output_path, llm_fn,
            batch_size=args.batch_size, n_batches=args.batches,
        )
        print(f"  Total: {total} items generated", file=sys.stderr)


if __name__ == "__main__":
    main()
