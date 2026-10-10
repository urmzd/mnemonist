#!/usr/bin/env bash
# Launch a sandboxed shell for the mnemonist showcase capture.
#
# teasr runs this as the terminal scene command (see ../teasr.toml) and types
# the demo into the shell it starts. With arguments it runs that single
# mnemonist command in the same sandbox instead. The demo runs real commands (learn,
# remember, consolidate, forget), so it must never touch a real store or pick
# up a stale binary:
#
#   - HOME points at a throwaway directory, so ~/.mnemonist is created fresh
#     on every run and deleted state can only ever be demo state.
#   - The binary built from this checkout goes first on PATH, so the capture
#     shows the CLI on this branch, not whatever version is installed.
#   - The prompt is fixed, so no username or hostname reaches the recording.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(dirname "$script_dir")"

if [[ -x "$repo_root/target/release/mnemonist" ]]; then
  bin_dir="$repo_root/target/release"
elif [[ -x "$repo_root/target/debug/mnemonist" ]]; then
  bin_dir="$repo_root/target/debug"
else
  echo "demo.sh: no built mnemonist binary; run \`cargo build --release -p mnemonist\` first" >&2
  exit 1
fi

# Reuse the already-downloaded embedding model instead of fetching it into the
# sandbox on every run. The cache is model weights only, never memory data.
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

demo="/tmp/mnemonist-demo"
rm -rf "$demo"
mkdir -p "$demo"
export HOME="$demo"
export PATH="$bin_dir:$PATH"
export PS1='mnemonist $ '
export BASH_SILENCE_DEPRECATION_WARNING=1
# Consolidation is its own step in the demo; keep it from also firing in the
# background and racing the recording.
export MNEMONIST_NO_AUTO_CONSOLIDATE=1

cd "$repo_root"

# With arguments, run that one mnemonist command (the cli-help scene).
if (($# > 0)); then
  exec mnemonist "$@"
fi
exec bash --noprofile --norc -i
