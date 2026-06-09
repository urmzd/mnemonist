#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["openai>=1.40"]
# ///
"""LongMemEval QA — Phase B (answer generation) + Phase C (LLM-judge scoring).

mnemonist-bench produces Phase A: per-question retrieved context as JSONL
(`QaContextRecord`). This script closes the loop the way the real LongMemEval
benchmark does:

  Phase B  `answer`  — feed each question + its retrieved context to a reader
                       LLM, producing a model answer (`QaAnswerRecord` JSONL,
                       compatible with `mnemonist-bench --qa-answers` for the
                       substring baseline).
  Phase C  `judge`   — score each answer against the gold answer with a strong
                       LLM judge (GPT-4o class), faithful to the LongMemEval
                       methodology (general correctness + temporal / knowledge-
                       update / abstention handling). Emits overall and
                       per-question-type accuracy.

  `all` runs both phases.

Usage:
  export OPENAI_API_KEY=...
  uv run scripts/longmemeval_qa.py answer  --context context.jsonl --out answers.jsonl
  uv run scripts/longmemeval_qa.py judge   --answers answers.jsonl --out judged.jsonl --report report.json
  uv run scripts/longmemeval_qa.py all     --context context.jsonl --report report.json \
        --reader-model gpt-4o-mini --judge-model gpt-4o
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI


def chat_with_retry(client: OpenAI, *, max_attempts: int = 8, **kwargs):
    """Call chat.completions with exponential backoff + jitter on rate limits /
    transient errors. Raises the last exception only if all attempts fail."""
    delay = 2.0
    last = None
    for attempt in range(max_attempts):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as e:  # noqa: BLE001
            last = e
            msg = str(e).lower()
            transient = (
                "429" in msg or "rate limit" in msg or "timeout" in msg
                or "overloaded" in msg or "500" in msg or "502" in msg
                or "503" in msg or "504" in msg or "connection" in msg
            )
            if not transient or attempt == max_attempts - 1:
                raise
            time.sleep(delay + random.uniform(0, delay))
            delay = min(delay * 2, 60)
    raise last

# ── Prompts ──────────────────────────────────────────────────────────────────

READER_SYSTEM = (
    "You are answering questions about a user based on excerpts from their past "
    "conversations with an AI assistant. Use ONLY the information in the provided "
    "context. If the answer cannot be found in the context, reply exactly with: "
    "I don't know.\n"
    "Answer concisely — give just the fact, name, number, date, or short phrase "
    "that the question asks for. Do not explain your reasoning."
)

JUDGE_SYSTEM = (
    "You are a strict evaluator for a long-term-memory question-answering benchmark. "
    "You will be given a question, the reference (correct) answer, and a model's "
    "response. Decide whether the model's response is correct.\n"
    "The response is CORRECT if it conveys the same key information as the reference "
    "answer; it may be phrased differently, be more verbose, or include correct extra "
    "detail. The response is INCORRECT if it contradicts the reference, omits the "
    "requested fact, or says it does not know when the reference provides an answer.\n"
    'Respond with a JSON object: {"correct": true} or {"correct": false}.'
)

# Per-type judge guidance appended to the user message (faithful to LongMemEval).
TYPE_GUIDANCE = {
    "temporal-reasoning": (
        " This is a temporal-reasoning question: the dates, durations, ordering, or "
        "time spans in the response must match the reference exactly to be correct."
    ),
    "knowledge-update": (
        " This is a knowledge-update question: the response must reflect the most "
        "recent / updated information, not superseded earlier information."
    ),
    "single-session-preference": (
        " This is a preference question: the response is correct if it captures the "
        "user's stated preference, even if phrased differently."
    ),
}

ABSTENTION_MARKERS = (
    "i don't know",
    "i do not know",
    "not mentioned",
    "no information",
    "cannot answer",
    "can't answer",
    "i don't have",
    "i do not have",
    "not enough information",
    "no relevant information",
    "isn't mentioned",
    "is not mentioned",
)


def is_abstention(text: str) -> bool:
    low = text.lower()
    return any(m in low for m in ABSTENTION_MARKERS)


def is_abstention_question(rec: dict) -> bool:
    """LongMemEval abstention questions have no answerable gold (the info was never
    given) and conventionally carry an `_abs` suffix in the question_id."""
    qid = (rec.get("question_id") or "")
    gold = rec.get("gold_answer")
    return qid.endswith("_abs") or gold in (None, "", [])


# ── Phase B: answer generation ────────────────────────────────────────────────

def build_context(rec: dict, max_chars: int) -> str:
    chunks = rec.get("retrieved_context") or []
    parts = []
    used = 0
    for i, c in enumerate(chunks, 1):
        block = f"[Conversation {i}]\n{c}\n"
        if used + len(block) > max_chars and parts:
            break
        parts.append(block)
        used += len(block)
    return "\n".join(parts) if parts else "(no context retrieved)"


def answer_one(client: OpenAI, rec: dict, model: str, max_chars: int) -> dict:
    context = build_context(rec, max_chars)
    today = rec.get("question_date")
    today_line = f"Today's date is {today}.\n" if today else ""
    user = f"{today_line}Context:\n\n{context}\n\nQuestion: {rec['question']}\nAnswer:"
    try:
        resp = chat_with_retry(
            client,
            model=model,
            messages=[
                {"role": "system", "content": READER_SYSTEM},
                {"role": "user", "content": user},
            ],
            temperature=0.0,
            max_tokens=256,
        )
        model_answer = (resp.choices[0].message.content or "").strip()
    except Exception as e:  # noqa: BLE001 — record failures, don't crash the batch
        model_answer = f"[ERROR: {e}]"
    return {
        "question_id": rec.get("question_id"),
        "question_type": rec.get("question_type"),
        "question": rec["question"],
        "gold_answer": rec.get("gold_answer"),
        "model_answer": model_answer,
        # carried through for richer reporting (ignored by the Rust scorer)
        "retrieval_hit": rec.get("retrieval_hit"),
    }


# ── Phase C: judging ──────────────────────────────────────────────────────────

def judge_one(client: OpenAI, rec: dict, model: str) -> dict:
    ans = rec.get("model_answer", "")
    out = dict(rec)

    # Abstention questions: correct iff the model declined to answer.
    if is_abstention_question(rec):
        out["correct"] = is_abstention(ans)
        out["judge_method"] = "abstention"
        return out

    # Guard: an empty/errored answer is wrong.
    if not ans or ans.startswith("[ERROR"):
        out["correct"] = False
        out["judge_method"] = "empty"
        return out

    guidance = TYPE_GUIDANCE.get(rec.get("question_type") or "", "")
    user = (
        f"Question: {rec['question']}\n"
        f"Reference answer: {rec.get('gold_answer')}\n"
        f"Model response: {ans}\n"
        f"{guidance}\n\n"
        'Is the model response correct? Respond with JSON: {"correct": true|false}.'
    )
    try:
        resp = chat_with_retry(
            client,
            model=model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": user},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
            max_tokens=16,
        )
        verdict = json.loads(resp.choices[0].message.content or "{}")
        out["correct"] = bool(verdict.get("correct", False))
        out["judge_method"] = "llm"
    except Exception as e:  # noqa: BLE001
        # Do NOT count an ungraded answer as wrong — mark it for re-run.
        out["correct"] = None
        out["judge_method"] = f"error:{e}"
    return out


# ── Batch driver ──────────────────────────────────────────────────────────────

def run_batch(fn, items, concurrency, label):
    results = [None] * len(items)
    done = 0
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = {ex.submit(fn, i, rec): idx for idx, (i, rec) in enumerate(items)}
        for fut in as_completed(futs):
            idx = futs[fut]
            results[idx] = fut.result()
            done += 1
            if done % 25 == 0 or done == len(items):
                print(f"  {label}: {done}/{len(items)}", file=sys.stderr)
    return results


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, records):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def wilson_interval(successes: int, n: int, z: float = 1.96) -> list[float]:
    """Wilson score interval [lo, hi] for a binomial proportion.

    Mirrors mnemonist-core evals::search::wilson_interval; n == 0 returns the
    uninformative [0, 1]."""
    if n == 0:
        return [0.0, 1.0]
    p = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    margin = (z / denom) * (p * (1 - p) / n + z2 / (4 * n * n)) ** 0.5
    return [max(center - margin, 0.0), min(center + margin, 1.0)]


def report(records: list[dict]) -> dict:
    per_type = defaultdict(lambda: {"correct": 0, "graded": 0, "total": 0, "retrieval_hit": 0})
    total_correct = 0
    total_graded = 0
    total_hit = 0
    ungraded = 0
    for r in records:
        t = r.get("question_type") or "unknown"
        per_type[t]["total"] += 1
        c = r.get("correct")
        if c is None:
            ungraded += 1
        else:
            per_type[t]["graded"] += 1
            total_graded += 1
            if c:
                per_type[t]["correct"] += 1
                total_correct += 1
        if r.get("retrieval_hit"):
            per_type[t]["retrieval_hit"] += 1
            total_hit += 1
    n = len(records)
    return {
        "n_questions": n,
        "n_graded": total_graded,
        "n_ungraded": ungraded,
        "n_correct": total_correct,
        "overall_accuracy": total_correct / total_graded if total_graded else 0.0,
        "overall_accuracy_ci95": wilson_interval(total_correct, total_graded),
        "retrieval_recall_any_at_k": total_hit / n if n else 0.0,
        "scoring_method": "llm_judge",
        "per_type": sorted(
            (
                {
                    "question_type": t,
                    "accuracy": v["correct"] / v["graded"] if v["graded"] else 0.0,
                    "accuracy_ci95": wilson_interval(v["correct"], v["graded"]),
                    "retrieval_recall": v["retrieval_hit"] / v["total"] if v["total"] else 0.0,
                    "count": v["total"],
                }
                for t, v in per_type.items()
            ),
            key=lambda x: x["question_type"],
        ),
    }


def print_report(rep: dict):
    print("\n═══ LongMemEval QA — LLM-judge results ═══")
    print(f"  questions:        {rep['n_questions']} ({rep.get('n_graded', rep['n_questions'])} graded, {rep.get('n_ungraded', 0)} ungraded)")
    ci = rep.get("overall_accuracy_ci95", [0.0, 1.0])
    print(
        f"  overall accuracy: {rep['overall_accuracy'] * 100:.1f}%  "
        f"[95% CI {ci[0] * 100:.1f}–{ci[1] * 100:.1f}]  (of graded)"
    )
    print(f"  retrieval recall: {rep['retrieval_recall_any_at_k'] * 100:.1f}%")
    print(f"  {'type':<28} {'acc':>7} {'95% CI':>14} {'recall':>8} {'n':>5}")
    for t in rep["per_type"]:
        tci = t.get("accuracy_ci95", [0.0, 1.0])
        print(
            f"  {t['question_type']:<28} {t['accuracy']*100:>6.1f}% "
            f"[{tci[0]*100:>5.1f}–{tci[1]*100:>5.1f}] "
            f"{t['retrieval_recall']*100:>7.1f}% {t['count']:>5}"
        )


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="LongMemEval QA answer + judge harness")
    sub = p.add_subparsers(dest="cmd", required=True)

    pa = sub.add_parser("answer", help="Phase B: generate answers from retrieved context")
    pa.add_argument("--context", required=True)
    pa.add_argument("--out", required=True)
    pa.add_argument("--reader-model", default="gpt-4o-mini")
    pa.add_argument("--concurrency", type=int, default=4)
    pa.add_argument("--max-context-chars", type=int, default=24000)
    pa.add_argument("--limit", type=int, default=0, help="cap #questions (0 = all)")

    pj = sub.add_parser("judge", help="Phase C: judge answers against gold")
    pj.add_argument("--answers", required=True)
    pj.add_argument("--out", required=True)
    pj.add_argument("--report", default="")
    pj.add_argument("--judge-model", default="gpt-4o")
    pj.add_argument("--concurrency", type=int, default=4)

    pall = sub.add_parser("all", help="Phase B + C")
    pall.add_argument("--context", required=True)
    pall.add_argument("--answers-out", default="answers.jsonl")
    pall.add_argument("--judged-out", default="judged.jsonl")
    pall.add_argument("--report", default="report.json")
    pall.add_argument("--reader-model", default="gpt-4o-mini")
    pall.add_argument("--judge-model", default="gpt-4o")
    pall.add_argument("--concurrency", type=int, default=4)
    pall.add_argument("--max-context-chars", type=int, default=24000)
    pall.add_argument("--limit", type=int, default=0)

    args = p.parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY not set")
    client = OpenAI(max_retries=8, timeout=90.0)

    if args.cmd == "answer":
        recs = load_jsonl(args.context)
        if args.limit:
            recs = recs[: args.limit]
        items = list(enumerate(recs))
        out = run_batch(
            lambda i, r: answer_one(client, r, args.reader_model, args.max_context_chars),
            items, args.concurrency, "answer",
        )
        write_jsonl(args.out, out)
        print(f"wrote {len(out)} answers to {args.out}", file=sys.stderr)

    elif args.cmd == "judge":
        recs = load_jsonl(args.answers)
        items = list(enumerate(recs))
        out = run_batch(
            lambda i, r: judge_one(client, r, args.judge_model),
            items, args.concurrency, "judge",
        )
        write_jsonl(args.out, out)
        rep = report(out)
        print_report(rep)
        if args.report:
            with open(args.report, "w") as f:
                json.dump(rep, f, indent=2)

    elif args.cmd == "all":
        recs = load_jsonl(args.context)
        if args.limit:
            recs = recs[: args.limit]
        items = list(enumerate(recs))
        answers = run_batch(
            lambda i, r: answer_one(client, r, args.reader_model, args.max_context_chars),
            items, args.concurrency, "answer",
        )
        write_jsonl(args.answers_out, answers)
        items2 = list(enumerate(answers))
        judged = run_batch(
            lambda i, r: judge_one(client, r, args.judge_model),
            items2, args.concurrency, "judge",
        )
        write_jsonl(args.judged_out, judged)
        rep = report(judged)
        print_report(rep)
        with open(args.report, "w") as f:
            json.dump(rep, f, indent=2)


if __name__ == "__main__":
    main()
