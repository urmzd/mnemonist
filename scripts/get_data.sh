#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${1:-data}"
BASE_URL="https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main"

FILES=(
  "longmemeval_s_cleaned.json"
  "longmemeval_m_cleaned.json"
  "longmemeval_oracle.json"
)

mkdir -p "$DATA_DIR"

for file in "${FILES[@]}"; do
  dest="$DATA_DIR/$file"
  if [ -f "$dest" ]; then
    echo "  skip $file (already exists)"
  else
    echo "  downloading $file..."
    curl -fSL "$BASE_URL/$file" -o "$dest"
  fi
done

echo "done — $(ls -1 "$DATA_DIR"/*.json 2>/dev/null | wc -l | tr -d ' ') files in $DATA_DIR/"
