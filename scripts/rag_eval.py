#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Code-retrieval RAG evaluation over real repositories.

Measures whether `mnemonist learn` + `mnemonist remember` surface the right
source files for natural-language developer queries. This is the code-RAG
counterpart to the conversational LongMemEval suite.

Isolation: each run uses a throwaway $HOME so the real ~/.mnemonist is never
touched, while HF_HOME points at the real model cache to avoid re-downloading
the embedding model. Source repos are read-only (learn writes only under the
isolated storage root).

Gold format: one JSONL file per repo under --gold-dir, named <repo>.jsonl, each
line: {"query": "...", "gold_files": ["path/rel/to/repo", ...], "why": "..."}.

Metrics (file-level, matching crates/mnemonist-core/src/evals/search.rs):
  recall_any@k  — >=1 gold file in top-k distinct retrieved files
  recall_all@k  — all gold files in top-k
  MRR           — 1 / rank of first gold file
  precision@5   — |gold ∩ top5| / 5

Usage:
  uv run scripts/rag_eval.py \
      --gold-dir docs/benchmarks/rag_gold \
      --repos-root /Users/urmzd/github \
      --out docs/benchmarks/rag_results.json \
      --md  docs/benchmarks/rag_results.md
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

KS = [1, 3, 5, 10]
LINE_SUFFIX = re.compile(r":\d+:\d+$")  # strip ":start:end" from "file:start:end"


def run(cmd: list[str], env: dict, cwd: str | None = None) -> str:
    res = subprocess.run(cmd, env=env, cwd=cwd, capture_output=True, text=True)
    if res.returncode != 0:
        sys.stderr.write(f"  ! command failed ({res.returncode}): {' '.join(cmd)}\n{res.stderr[:500]}\n")
    return res.stdout


def isolated_env(eval_home: str, hf_home: str) -> dict:
    env = dict(os.environ)
    env["HOME"] = eval_home
    if hf_home:
        env["HF_HOME"] = hf_home
    return env


def retrieved_files(stdout: str) -> list[str]:
    """Parse `remember --format json` stdout -> ranked list of distinct files."""
    try:
        doc = json.loads(stdout)
    except json.JSONDecodeError:
        return []
    mems = (doc.get("data") or {}).get("memories") or []
    files: list[str] = []
    seen = set()
    for m in mems:
        raw = m.get("file") or m.get("name") or ""
        path = LINE_SUFFIX.sub("", raw)
        if path and path not in seen:
            seen.add(path)
            files.append(path)
    return files


def score_query(retrieved: list[str], gold: list[str]) -> dict:
    gold_set = set(gold)
    out = {}
    for k in KS:
        topk = set(retrieved[:k])
        out[f"recall_any@{k}"] = 1.0 if topk & gold_set else 0.0
        out[f"recall_all@{k}"] = 1.0 if gold_set and gold_set <= topk else 0.0
    # MRR over first gold file
    mrr = 0.0
    for i, f in enumerate(retrieved, 1):
        if f in gold_set:
            mrr = 1.0 / i
            break
    out["mrr"] = mrr
    top5 = retrieved[:5]
    out["precision@5"] = sum(1 for f in top5 if f in gold_set) / 5.0
    return out


def mean(rows: list[dict], key: str) -> float:
    vals = [r[key] for r in rows if key in r]
    return sum(vals) / len(vals) if vals else 0.0


def eval_repo(repo: str, repo_path: str, gold: list[dict], args, env: dict) -> dict:
    print(f"\n=== {repo} ({len(gold)} queries) ===", file=sys.stderr)
    if not args.no_learn:
        print(f"  learning {repo_path} ...", file=sys.stderr)
        run([args.bin, "learn", repo_path, "--root", repo_path, "--quiet", "--format", "json"], env)

    per_query = []
    for q in gold:
        out = run(
            [args.bin, "remember", q["query"], "--root", repo_path,
             "--level", "project", "--budget", str(args.budget), "--format", "json", "--quiet"],
            env,
        )
        files = retrieved_files(out)
        sc = score_query(files, q["gold_files"])
        sc["query"] = q["query"]
        sc["gold_files"] = q["gold_files"]
        sc["top5"] = files[:5]
        per_query.append(sc)

    agg = {k: mean(per_query, k) for k in
           [f"recall_any@{k}" for k in KS] + [f"recall_all@{k}" for k in KS] + ["mrr", "precision@5"]}
    agg["n_queries"] = len(per_query)
    return {"repo": repo, "aggregate": agg, "per_query": per_query}


def main():
    p = argparse.ArgumentParser(description="Code-retrieval RAG eval over real repos")
    p.add_argument("--gold-dir", required=True)
    p.add_argument("--repos-root", required=True)
    p.add_argument("--bin", default="mnemonist", help="mnemonist binary (PATH name or path)")
    p.add_argument("--eval-home", default="/tmp/mnem-rageval")
    p.add_argument("--hf-home", default=os.path.expanduser("~/.cache/huggingface"))
    p.add_argument("--budget", type=int, default=50000, help="char budget (large => up to top-10 not truncated)")
    p.add_argument("--no-learn", action="store_true", help="skip re-learn (reuse existing indices)")
    p.add_argument("--out", default="")
    p.add_argument("--md", default="")
    args = p.parse_args()

    env = isolated_env(args.eval_home, args.hf_home)
    Path(args.eval_home).mkdir(parents=True, exist_ok=True)

    gold_files = sorted(Path(args.gold_dir).glob("*.jsonl"))
    if not gold_files:
        sys.exit(f"no *.jsonl gold files in {args.gold_dir}")

    repos = []
    for gf in gold_files:
        repo = gf.stem
        repo_path = str(Path(args.repos_root) / repo)
        if not Path(repo_path).is_dir():
            sys.stderr.write(f"  skip {repo}: {repo_path} not found\n")
            continue
        gold = [json.loads(line) for line in gf.read_text().splitlines() if line.strip()]
        repos.append(eval_repo(repo, repo_path, gold, args, env))

    # Macro-average across repos (each repo weighted equally).
    metric_keys = [f"recall_any@{k}" for k in KS] + [f"recall_all@{k}" for k in KS] + ["mrr", "precision@5"]
    overall = {k: sum(r["aggregate"][k] for r in repos) / len(repos) for k in metric_keys}
    overall["n_repos"] = len(repos)
    overall["n_queries"] = sum(r["aggregate"]["n_queries"] for r in repos)

    result = {"overall_macro": overall, "repos": repos}

    # ── report ──
    print("\n══ Code-RAG retrieval (file-level) ══")
    hdr = f"{'repo':<12} {'n':>3} {'R@1':>6} {'R@3':>6} {'R@5':>6} {'R@10':>6} {'MRR':>6} {'P@5':>6}"
    print(hdr)
    for r in repos:
        a = r["aggregate"]
        print(f"{r['repo']:<12} {a['n_queries']:>3} "
              f"{a['recall_any@1']*100:>5.0f}% {a['recall_any@3']*100:>5.0f}% "
              f"{a['recall_any@5']*100:>5.0f}% {a['recall_any@10']*100:>5.0f}% "
              f"{a['mrr']:>6.3f} {a['precision@5']:>6.3f}")
    print(f"{'OVERALL':<12} {overall['n_queries']:>3} "
          f"{overall['recall_any@1']*100:>5.0f}% {overall['recall_any@3']*100:>5.0f}% "
          f"{overall['recall_any@5']*100:>5.0f}% {overall['recall_any@10']*100:>5.0f}% "
          f"{overall['mrr']:>6.3f} {overall['precision@5']:>6.3f}")

    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2))
        print(f"\nwrote {args.out}", file=sys.stderr)
    if args.md:
        lines = ["| repo | n | recall@1 | recall@3 | recall@5 | recall@10 | MRR | precision@5 |",
                 "|---|---|---|---|---|---|---|---|"]
        for r in repos + [{"repo": "**overall (macro)**", "aggregate": overall}]:
            a = r["aggregate"]
            lines.append(
                f"| {r['repo']} | {a.get('n_queries','')} | "
                f"{a['recall_any@1']*100:.0f}% | {a['recall_any@3']*100:.0f}% | "
                f"{a['recall_any@5']*100:.0f}% | {a['recall_any@10']*100:.0f}% | "
                f"{a['mrr']:.3f} | {a['precision@5']:.3f} |")
        Path(args.md).write_text("\n".join(lines) + "\n")
        print(f"wrote {args.md}", file=sys.stderr)


if __name__ == "__main__":
    main()
