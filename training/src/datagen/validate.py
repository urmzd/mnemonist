"""Validate generated JSONL training data for quality and format."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


def validate_code_pairs(path: Path) -> dict:
    """Validate code_pairs.jsonl."""
    errors = []
    stats = Counter()
    required_fields = {"language", "code", "description"}

    for i, line in enumerate(path.open(), 1):
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            errors.append(f"Line {i}: invalid JSON")
            continue

        missing = required_fields - set(obj.keys())
        if missing:
            errors.append(f"Line {i}: missing fields: {missing}")
            continue

        stats[obj["language"]] += 1

        if len(obj["code"]) < 20:
            errors.append(f"Line {i}: code too short ({len(obj['code'])} chars)")
        if len(obj["description"]) < 10:
            errors.append(f"Line {i}: description too short")

    return {"total": sum(stats.values()), "by_language": dict(stats), "errors": errors[:20]}


def validate_memory_files(path: Path) -> dict:
    """Validate memory_files.jsonl."""
    errors = []
    stats = Counter()
    required_fields = {"type", "name", "description", "body"}
    valid_types = {"user", "feedback", "project", "reference"}

    for i, line in enumerate(path.open(), 1):
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            errors.append(f"Line {i}: invalid JSON")
            continue

        missing = required_fields - set(obj.keys())
        if missing:
            errors.append(f"Line {i}: missing fields: {missing}")
            continue

        if obj["type"] not in valid_types:
            errors.append(f"Line {i}: invalid type: {obj['type']}")

        stats[obj["type"]] += 1

        if len(obj["description"]) > 150:
            errors.append(f"Line {i}: description too long ({len(obj['description'])} chars)")

    return {"total": sum(stats.values()), "by_type": dict(stats), "errors": errors[:20]}


def validate_temporal_scenarios(path: Path) -> dict:
    """Validate temporal_scenarios.jsonl."""
    errors = []
    count = 0

    for i, line in enumerate(path.open(), 1):
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            errors.append(f"Line {i}: invalid JSON")
            continue

        if "query" not in obj:
            errors.append(f"Line {i}: missing query")
        if "candidates" not in obj or len(obj.get("candidates", [])) < 2:
            errors.append(f"Line {i}: need at least 2 candidates")

        for j, c in enumerate(obj.get("candidates", [])):
            if "temporal_relevance" not in c:
                errors.append(f"Line {i}, candidate {j}: missing temporal_relevance")
            elif not 0 <= c["temporal_relevance"] <= 1:
                errors.append(f"Line {i}, candidate {j}: temporal_relevance out of [0,1]")

        count += 1

    return {"total": count, "errors": errors[:20]}
