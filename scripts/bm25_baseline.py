#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""BM25 lexical baseline for the LongMemEval retrieval construct (Exp 1).

Runs plain Okapi BM25 over the exact same setup as Experiment 1: per-question
haystacks (~48 sessions each), documents are the concatenated **user turns** of
each session (the retrieval representation in evals/longmemeval.rs), gold is
`answer_session_ids`. No embeddings, no index — just term statistics.

The point of this baseline is construct validity: if BM25 lands within a few
points of the vector numbers, the per-question-haystack task is near-saturated
and recall_any@5 on it does not discriminate between retrieval systems.

Metrics mirror crates/mnemonist-core/src/evals/search.rs exactly:
  recall_any@k — >=1 gold session in top-k (empty-gold queries count as misses)
  recall_all@k — every gold session in top-k (empty gold -> miss)
  MRR          — 1 / rank of first gold over the retrieved list (top-10)
Intervals are 95% Wilson, matching the harness.

Usage:
  uv run scripts/bm25_baseline.py \
      --dataset data/longmemeval_s_cleaned.json \
      --out docs/benchmarks/bm25_baseline.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter

K1 = 1.5
B = 0.75
TOKEN_RE = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def bm25_rank(query_tokens: list[str], docs: list[list[str]], top_k: int) -> list[int]:
    """Rank document indices by Okapi BM25 score for the query."""
    n = len(docs)
    avgdl = sum(len(d) for d in docs) / n if n else 0.0
    doc_freqs = [Counter(d) for d in docs]
    # Document frequency per query term.
    df = {
        t: sum(1 for f in doc_freqs if t in f)
        for t in set(query_tokens)
    }
    idf = {
        t: math.log((n - dft + 0.5) / (dft + 0.5) + 1.0)
        for t, dft in df.items()
    }
    scores = []
    for i, freqs in enumerate(doc_freqs):
        dl = len(docs[i])
        s = 0.0
        for t in query_tokens:
            tf = freqs.get(t, 0)
            if tf == 0:
                continue
            denom = tf + K1 * (1 - B + B * dl / avgdl)
            s += idf[t] * tf * (K1 + 1) / denom
        scores.append((s, i))
    scores.sort(key=lambda x: (-x[0], x[1]))
    return [i for _, i in scores[:top_k]]


def wilson95(hits: int, n: int) -> list[float]:
    if n == 0:
        return [0.0, 0.0]
    z = 1.959963984540054
    p = hits / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [max(0.0, center - margin), min(1.0, center + margin)]


def user_turns_document(turns: list[dict]) -> str:
    return " ".join(t["content"] for t in turns if t.get("role") == "user")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    with open(args.dataset) as f:
        entries = json.load(f)

    n_scored = 0
    any5 = all5 = any10 = all10 = 0
    mrr_sum = 0.0
    haystack_total = 0

    for idx, e in enumerate(entries):
        ids = e["haystack_session_ids"]
        sessions = e["haystack_sessions"]
        docs, valid_ids = [], []
        for sid, turns in zip(ids, sessions):
            docs.append(tokenize(user_turns_document(turns)))
            valid_ids.append(sid)
        if not docs:
            continue
        haystack_total += len(docs)

        top = bm25_rank(tokenize(e["question"]), docs, 10)
        retrieved = [valid_ids[i] for i in top]
        gold = set(e["answer_session_ids"])

        top5, top10 = set(retrieved[:5]), set(retrieved[:10])
        any5 += bool(gold & top5)
        any10 += bool(gold & top10)
        all5 += bool(gold) and gold <= top5
        all10 += bool(gold) and gold <= top10
        mrr_sum += next(
            (1.0 / (r + 1) for r, sid in enumerate(retrieved) if sid in gold), 0.0
        )
        n_scored += 1
        if (idx + 1) % 50 == 0 or idx + 1 == len(entries):
            print(f"  [{idx + 1}/{len(entries)}]", flush=True)

    result = {
        "method": "bm25_okapi",
        "params": {"k1": K1, "b": B, "tokenizer": r"\w+ lowercase"},
        "construct": "per-question haystack, user-turns-only documents (matches Exp 1)",
        "n_queries": n_scored,
        "avg_haystack_size": haystack_total / n_scored if n_scored else 0.0,
        "recall_any_at_5": any5 / n_scored,
        "recall_any_at_5_ci95": wilson95(any5, n_scored),
        "recall_all_at_5": all5 / n_scored,
        "recall_all_at_5_ci95": wilson95(all5, n_scored),
        "recall_any_at_10": any10 / n_scored,
        "recall_any_at_10_ci95": wilson95(any10, n_scored),
        "recall_all_at_10": all10 / n_scored,
        "recall_all_at_10_ci95": wilson95(all10, n_scored),
        "mrr": mrr_sum / n_scored,
    }
    print(json.dumps(result, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
            f.write("\n")


if __name__ == "__main__":
    main()
