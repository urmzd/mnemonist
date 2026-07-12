#!/usr/bin/env bash
set -uo pipefail

# ── mnemonist full system validation ─────────────────────────────────────────────
# Runs every command against a clean, isolated environment and reports
# pass/fail with timing for each operation.

MNEMONIST="${MNEMONIST_BIN:-$(dirname "$0")/../target/release/mnemonist}"

# ── Theme ────────────────────────────────────────────────────────────────────
BOLD="\033[1m"
DIM="\033[2m"
RESET="\033[0m"
GREEN="\033[32m"
RED="\033[31m"
CYAN="\033[36m"
WHITE="\033[97m"
BAR_FG="\033[38;5;75m"

PASS_COUNT=0
FAIL_COUNT=0
TOTAL_MS=0
declare -a RESULTS=()

# ── Helpers ──────────────────────────────────────────────────────────────────

now_ms() { python3 -c 'import time; print(int(time.time()*1000))'; }

# Shared output variable — avoids subshell variable loss
OUT=""

# run NAME CMD...   — expect success
run() {
  local name="$1"; shift
  local t0; t0=$(now_ms)
  local rc=0
  OUT=$("$@" 2>/dev/null) || rc=$?
  local t1; t1=$(now_ms)
  local ms=$((t1 - t0))
  TOTAL_MS=$((TOTAL_MS + ms))
  if [ "$rc" -eq 0 ]; then
    PASS_COUNT=$((PASS_COUNT + 1)); RESULTS+=("PASS|${ms}|${name}")
  else
    FAIL_COUNT=$((FAIL_COUNT + 1)); RESULTS+=("FAIL|${ms}|${name}")
  fi
}

# run_fail NAME CMD...   — expect nonzero exit
run_fail() {
  local name="$1"; shift
  local t0; t0=$(now_ms)
  local rc=0
  OUT=$("$@" 2>/dev/null) || rc=$?
  local t1; t1=$(now_ms)
  local ms=$((t1 - t0))
  TOTAL_MS=$((TOTAL_MS + ms))
  if [ "$rc" -ne 0 ]; then
    PASS_COUNT=$((PASS_COUNT + 1)); RESULTS+=("PASS|${ms}|${name}")
  else
    FAIL_COUNT=$((FAIL_COUNT + 1)); RESULTS+=("FAIL|${ms}|${name}")
  fi
}

# eq PY_EXPR EXPECTED LABEL
eq() {
  local expr="$1" expected="$2" label="$3"
  local actual
  actual=$(echo "$OUT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(json.dumps($expr))" 2>/dev/null) || actual="__PARSE_ERROR__"
  if [ "$actual" = "$expected" ]; then
    PASS_COUNT=$((PASS_COUNT + 1)); RESULTS+=("PASS|0|${label}")
  else
    FAIL_COUNT=$((FAIL_COUNT + 1)); RESULTS+=("FAIL|0|${label} (expected $expected, got $actual)")
  fi
}

# ok PY_BOOL_EXPR LABEL
ok() {
  local expr="$1" label="$2"
  local val
  val=$(echo "$OUT" | python3 -c "import sys,json; d=json.load(sys.stdin); print('yes' if ($expr) else 'no')" 2>/dev/null) || val="no"
  if [ "$val" = "yes" ]; then
    PASS_COUNT=$((PASS_COUNT + 1)); RESULTS+=("PASS|0|${label}")
  else
    FAIL_COUNT=$((FAIL_COUNT + 1)); RESULTS+=("FAIL|0|${label}")
  fi
}

cleanup() {
  [ -n "${SERVER_PID:-}" ] && kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
  rm -rf "$TMPDIR"
}
trap cleanup EXIT

# ── Setup ────────────────────────────────────────────────────────────────────

TMPDIR=$(mktemp -d)
export HOME="$TMPDIR/home"
# Keep runs deterministic: no detached background consolidation mid-validation
export MNEMONIST_NO_AUTO_CONSOLIDATE=1
mkdir -p "$HOME"
PROJECT="$TMPDIR/myproject"
mkdir -p "$PROJECT"
REPO_ROOT="$(cd "$(dirname "$0")/.."; pwd)"
SERVER_PID=""

echo ""
echo -e "${BOLD}${CYAN}  mnemonist${RESET}${BOLD} system validation${RESET}"
echo -e "${DIM}  ────────────────────────────────────────${RESET}"
echo ""

# ── 1. Config ────────────────────────────────────────────────────────────────
echo -e "${WHITE}  config${RESET}"

run "config init" "$MNEMONIST" config init
eq "d['data']['action']" '"created"' "config init: action is created"

run "config get" "$MNEMONIST" config get embedding.model
eq "d['data']['value']" '"sentence-transformers/all-MiniLM-L6-v2"' "config get: default model"

run "config set" "$MNEMONIST" config set embedding.model "test-embed"
eq "d['data']['value']" '"test-embed"' "config set: value updated"

run "config get verify" "$MNEMONIST" config get embedding.model
eq "d['data']['value']" '"test-embed"' "config set: persisted"

# Restore for embedding to work
"$MNEMONIST" config set embedding.model "sentence-transformers/all-MiniLM-L6-v2" >/dev/null 2>/dev/null

run "config show" "$MNEMONIST" config show
ok "len(d['data']['config']) > 100" "config show: non-empty"

run "config path" "$MNEMONIST" config path
ok "'mnemonist.toml' in d['data']['path']" "config path: ends in mnemonist.toml"

# ── 2. Remember (store) ─────────────────────────────────────────────────────
echo -e "${WHITE}  remember${RESET}"

run "remember feedback" "$MNEMONIST" remember "always write tests" --root "$PROJECT"
eq "d['data']['action']" '"created"' "remember: action is created"
ok "d['data']['file'].startswith('feedback_')" "remember: file prefix"

run "remember user" "$MNEMONIST" remember "prefers rust" -t user --name "lang-pref" --root "$PROJECT"
eq "d['data']['file']" '"user_lang-pref.md"' "remember user: correct filename"

run "remember project" "$MNEMONIST" remember "merge freeze march 5" -t project --name "freeze" --root "$PROJECT"
eq "d['data']['file']" '"project_freeze.md"' "remember project: correct filename"

run "remember reference" "$MNEMONIST" remember "see Linear INGEST board" -t reference --name "linear-board" --root "$PROJECT"
eq "d['data']['file']" '"reference_linear-board.md"' "remember reference: correct filename"

# Upsert same name
run "remember upsert" "$MNEMONIST" remember "prefers rust and go" -t user --name "lang-pref" --root "$PROJECT"
eq "d['data']['action']" '"updated"' "remember upsert: action is updated"

# Stdin
run "remember stdin" sh -c "echo '{\"type\":\"feedback\",\"name\":\"stdin-mem\",\"description\":\"from stdin\",\"body\":\"body content\",\"level\":\"project\"}' | '$MNEMONIST' remember ignored --stdin --root '$PROJECT'"
eq "d['data']['file']" '"feedback_stdin-mem.md"' "remember stdin: correct file"

# ── 3. Remember --defer (inbox) ──────────────────────────────────────────────
echo -e "${WHITE}  remember --defer${RESET}"

run "defer 1" "$MNEMONIST" remember --defer "check logging" --root "$PROJECT"
eq "d['data']['inbox_size']" '1' "defer: inbox_size is 1"

run "defer 2" "$MNEMONIST" remember --defer "review auth middleware" --root "$PROJECT"
eq "d['data']['inbox_size']" '2' "defer: inbox_size is 2"

# Fill inbox beyond capacity
for i in $(seq 3 13); do
  "$MNEMONIST" remember --defer "deferred item $i" --root "$PROJECT" >/dev/null 2>/dev/null
done
run "defer capacity" "$MNEMONIST" reflect --root "$PROJECT"
ok "d['data']['inbox']['size'] <= 10" "defer: respects capacity (<=10)"

run_fail "defer rejects stdin" "$MNEMONIST" remember --defer --stdin "point" --root "$PROJECT"
eq "d['ok']" 'false' "defer --stdin: rejected"

# ── 4. Recall ────────────────────────────────────────────────────────────────
echo -e "${WHITE}  recall${RESET}"

run "recall rust" "$MNEMONIST" recall "rust" --level project --root "$PROJECT"
ok "len(d['data']['memories']) >= 1" "recall: finds rust memories"
ok "d['data']['token_estimate'] >= 0" "recall: has token estimate"

run "recall obscure" "$MNEMONIST" recall "xyzzy999" --level project --root "$PROJECT"
ok "'memories' in d['data']" "recall: returns memories array"

# ── 5. Reflect───────────────────────────────────────────────────────────────
echo -e "${WHITE}  reflect${RESET}"

run "reflect" "$MNEMONIST" reflect --root "$PROJECT"
ok "len(d['data']['memories']) == 5" "reflect: 5 memories"
ok "d['data']['inbox']['size'] > 0" "reflect: inbox non-empty"

# ── 6. Learn─────────────────────────────────────────────────────────────────
echo -e "${WHITE}  learn${RESET}"

run "learn codebase" "$MNEMONIST" learn "$REPO_ROOT" --root "$PROJECT"
ok "d['data']['chunks'] > 0" "learn: extracted chunks"
ok "d['data']['files'] > 0" "learn: found files"

# ── 7. Consolidate───────────────────────────────────────────────────────────
echo -e "${WHITE}  consolidate${RESET}"

run "consolidate dry-run" "$MNEMONIST" consolidate --dry-run --root "$PROJECT"
eq "d['data']['dry_run']" 'true' "consolidate dry-run: flag set"
ok "d['data']['promoted'] > 0" "consolidate dry-run: would promote"

run "consolidate" "$MNEMONIST" consolidate --root "$PROJECT"
eq "d['data']['dry_run']" 'false' "consolidate: not dry run"
ok "d['data']['promoted'] > 0" "consolidate: promoted items"
eq "d['data']['decayed']" '0' "consolidate: no immediate decay (bug fix)"

# Verify inbox drained
run "post-consolidate reflect" "$MNEMONIST" reflect --root "$PROJECT"
eq "d['data']['inbox']['size']" '0' "consolidate: inbox drained"
ok "len(d['data']['memories']) > 5" "consolidate: memories grew"

# ── 8. Forget────────────────────────────────────────────────────────────────
echo -e "${WHITE}  forget${RESET}"

run "forget" "$MNEMONIST" forget feedback_stdin-mem.md --root "$PROJECT"
eq "d['data']['action']" '"forgotten"' "forget: action is forgotten"

run_fail "forget nonexistent" "$MNEMONIST" forget nonexistent.md --root "$PROJECT"
eq "d['ok']" 'false' "forget nonexistent: ok is false"

# ── 9. Consolidation lock ────────────────────────────────────────────────────
echo -e "${WHITE}  lock${RESET}"

MEM_DIR="$HOME/.mnemonist/myproject"
touch "$MEM_DIR/.consolidate.lock"
run "consolidate while locked" "$MNEMONIST" consolidate --root "$PROJECT"
eq "d['data']['skipped']" '"locked"' "lock: second run skips"
rm -f "$MEM_DIR/.consolidate.lock"

# ── Report ───────────────────────────────────────────────────────────────────

echo ""
echo -e "${DIM}  ────────────────────────────────────────────────────────────${RESET}"
echo ""

# Find max name length for alignment
MAX_NAME=0
for r in "${RESULTS[@]}"; do
  IFS='|' read -r status ms name <<< "$r"
  len=${#name}
  (( len > MAX_NAME )) && MAX_NAME=$len
done

# Calculate max ms for bar scaling
MAX_MS=1
for r in "${RESULTS[@]}"; do
  IFS='|' read -r status ms name <<< "$r"
  (( ms > MAX_MS )) && MAX_MS=$ms
done

BAR_WIDTH=24

for r in "${RESULTS[@]}"; do
  IFS='|' read -r status ms name <<< "$r"

  if [ "$status" = "PASS" ]; then
    icon="${GREEN}${RESET}"
  else
    icon="${RED}${RESET}"
  fi

  # Timing bar (only for bench'd items with ms > 0)
  bar=""
  timing=""
  if [ "$ms" -gt 0 ]; then
    filled=$(( (ms * BAR_WIDTH) / MAX_MS ))
    (( filled < 1 )) && filled=1
    empty=$((BAR_WIDTH - filled))
    bar="${BAR_FG}"
    for ((b=0; b<filled; b++)); do bar+="█"; done
    for ((b=0; b<empty; b++)); do bar+="░"; done
    bar+="${RESET}"
    timing="${DIM}${ms}ms${RESET}"
  fi

  printf "  %b %-${MAX_NAME}s  %b %b\n" "$icon" "$name" "$bar" "$timing"
done

echo ""
echo -e "${DIM}  ────────────────────────────────────────────────────────────${RESET}"

TOTAL=$((PASS_COUNT + FAIL_COUNT))
TOTAL_S=$(python3 -c "print(f'{$TOTAL_MS/1000:.2f}')")
if [ "$FAIL_COUNT" -eq 0 ]; then
  echo -e ""
  echo -e "  ${GREEN}${BOLD}${PASS_COUNT}/${TOTAL} passed${RESET}  ${DIM}in ${TOTAL_S}s${RESET}"
else
  echo -e ""
  echo -e "  ${GREEN}${PASS_COUNT} passed${RESET}  ${RED}${BOLD}${FAIL_COUNT} failed${RESET}  ${DIM}in ${TOTAL_S}s${RESET}"
fi
echo ""

exit "$FAIL_COUNT"
