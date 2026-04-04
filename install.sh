#!/usr/bin/env bash
set -euo pipefail

REPO="urmzd/mnemonist"
INSTALL_DIR="${INSTALL_DIR:-/usr/local/bin}"
BINARY="${BINARY:-mnemonist}"

usage() {
  cat <<EOF
Install mnemonist binaries from GitHub releases.

Usage: install.sh [OPTIONS]

Options:
  --binary <name>    Binary to install: mnemonist (default) or mnemonist-server
  --tag <tag>        Install a specific version (e.g. v0.1.0). Default: latest
  --target <triple>  Override target triple detection
  --dir <path>       Install directory (default: /usr/local/bin)
  --musl             Prefer musl over gnu on Linux
  -h, --help         Show this help

Examples:
  curl -fsSL https://raw.githubusercontent.com/$REPO/main/install.sh | bash
  curl -fsSL https://raw.githubusercontent.com/$REPO/main/install.sh | bash -s -- --binary mnemonist-server
  curl -fsSL https://raw.githubusercontent.com/$REPO/main/install.sh | bash -s -- --musl --tag v0.1.0
EOF
  exit 0
}

die() { echo "error: $1" >&2; exit 1; }

PREFER_MUSL=false
TAG=""
TARGET=""

while [ $# -gt 0 ]; do
  case "$1" in
    --binary)  BINARY="$2";      shift 2 ;;
    --tag)     TAG="$2";         shift 2 ;;
    --target)  TARGET="$2";      shift 2 ;;
    --dir)     INSTALL_DIR="$2"; shift 2 ;;
    --musl)    PREFER_MUSL=true; shift ;;
    -h|--help) usage ;;
    *)         die "unknown option: $1" ;;
  esac
done

detect_target() {
  local os arch triple

  os="$(uname -s)"
  arch="$(uname -m)"

  case "$os" in
    Linux)
      case "$arch" in
        x86_64)
          if [ "$PREFER_MUSL" = true ]; then
            triple="x86_64-unknown-linux-musl"
          else
            triple="x86_64-unknown-linux-gnu"
          fi
          ;;
        aarch64|arm64)
          if [ "$PREFER_MUSL" = true ]; then
            triple="aarch64-unknown-linux-musl"
          else
            triple="aarch64-unknown-linux-gnu"
          fi
          ;;
        *) die "unsupported architecture: $arch" ;;
      esac
      ;;
    Darwin)
      case "$arch" in
        x86_64)  triple="x86_64-apple-darwin" ;;
        arm64)   triple="aarch64-apple-darwin" ;;
        *)       die "unsupported architecture: $arch" ;;
      esac
      ;;
    MINGW*|MSYS*|CYGWIN*)
      triple="x86_64-pc-windows-msvc"
      ;;
    *) die "unsupported OS: $os" ;;
  esac

  echo "$triple"
}

if [ -z "$TARGET" ]; then
  TARGET="$(detect_target)"
fi

if [ -z "$TAG" ]; then
  TAG="$(curl -fsSL "https://api.github.com/repos/$REPO/releases/latest" | grep '"tag_name"' | cut -d'"' -f4)"
  [ -n "$TAG" ] || die "could not determine latest release"
fi

EXT=""
case "$TARGET" in
  *windows*) EXT=".exe" ;;
esac

ASSET="${BINARY}-${TARGET}${EXT}"
URL="https://github.com/$REPO/releases/download/${TAG}/${ASSET}"

echo "installing $BINARY $TAG ($TARGET)"
echo "  from: $URL"
echo "  to:   $INSTALL_DIR/$BINARY${EXT}"

TMPFILE="$(mktemp)"
trap 'rm -f "$TMPFILE"' EXIT

curl -fsSL -o "$TMPFILE" "$URL" || die "download failed — check that $TAG has a build for $TARGET"
chmod +x "$TMPFILE"

mkdir -p "$INSTALL_DIR"
mv "$TMPFILE" "$INSTALL_DIR/$BINARY${EXT}"

echo "done: $("$INSTALL_DIR/$BINARY${EXT}" --version 2>/dev/null || echo "$BINARY installed")"
