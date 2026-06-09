#!/bin/sh
# install.sh — Installs the mnemonist binary from GitHub releases.
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/urmzd/mnemonist/main/install.sh | sh
#
# Environment variables:
#   MNEMONIST_VERSION     — version to install (e.g. "v0.1.0"); defaults to latest
#   MNEMONIST_INSTALL_DIR — installation directory; defaults to $HOME/.local/bin
#   MNEMONIST_SHA256      — expected SHA256 checksum of the binary (hex string); skips verification if unset

set -eu

REPO="urmzd/mnemonist"

# curl with optional auth — uses GH_TOKEN or GITHUB_TOKEN if set.
gh_curl() {
    token="${GH_TOKEN:-${GITHUB_TOKEN:-}}"
    if [ -n "$token" ]; then
        curl -fsSL -H "Authorization: token $token" "$@"
    else
        curl -fsSL "$@"
    fi
}

main() {
    os=$(uname -s)
    arch=$(uname -m)

    case "$os" in
        Linux)
            case "$arch" in
                x86_64)  target="x86_64-unknown-linux-gnu" ;;
                aarch64) target="aarch64-unknown-linux-gnu" ;;
                *)       err "Unsupported Linux architecture: $arch" ;;
            esac
            ;;
        Darwin)
            case "$arch" in
                x86_64)  target="x86_64-apple-darwin" ;;
                arm64)   target="aarch64-apple-darwin" ;;
                *)       err "Unsupported macOS architecture: $arch" ;;
            esac
            ;;
        MINGW*|MSYS*|CYGWIN*|Windows_NT)
            err "Windows is not supported by this installer. Download a binary from https://github.com/$REPO/releases/latest"
            ;;
        *)
            err "Unsupported operating system: $os"
            ;;
    esac

    if [ -n "${MNEMONIST_VERSION:-}" ]; then
        tag="$MNEMONIST_VERSION"
    else
        tag=$(gh_curl "https://api.github.com/repos/$REPO/releases/latest" \
            | sed -n 's/.*"tag_name": *"\([^"]*\)".*/\1/p')
        if [ -z "$tag" ]; then
            err "Failed to fetch latest release tag"
        fi
    fi

    artifact="mnemonist-${target}"
    url="https://github.com/$REPO/releases/download/${tag}/${artifact}"

    install_dir="${MNEMONIST_INSTALL_DIR:-$HOME/.local/bin}"
    mkdir -p "$install_dir"

    echo "Downloading mnemonist $tag for $target..."
    gh_curl "$url" -o "$install_dir/mnemonist"

    if [ -n "${MNEMONIST_SHA256:-}" ]; then
        if command -v sha256sum >/dev/null 2>&1; then
            actual=$(sha256sum "$install_dir/mnemonist" | awk '{print $1}')
        elif command -v shasum >/dev/null 2>&1; then
            actual=$(shasum -a 256 "$install_dir/mnemonist" | awk '{print $1}')
        else
            err "sha256sum or shasum required for checksum verification"
        fi
        if [ "$actual" != "$MNEMONIST_SHA256" ]; then
            rm -f "$install_dir/mnemonist"
            err "SHA256 mismatch: expected $MNEMONIST_SHA256, got $actual"
        fi
        echo "SHA256 verified: $actual"
    fi

    chmod +x "$install_dir/mnemonist"

    echo "Installed mnemonist to $install_dir/mnemonist"

    case ":$PATH:" in
        *":$install_dir:"*) ;;
        *) add_to_path "$install_dir" ;;
    esac
}

add_to_path() {
    install_dir="$1"

    case "$(basename "$SHELL")" in
        zsh)  profile="$HOME/.zshrc" ;;
        bash)
            if [ -f "$HOME/.bashrc" ]; then
                profile="$HOME/.bashrc"
            else
                profile="$HOME/.profile"
            fi
            ;;
        fish) profile="$HOME/.config/fish/config.fish" ;;
        *)    profile="$HOME/.profile" ;;
    esac

    if [ "$(basename "$SHELL")" = "fish" ]; then
        if ! grep -q "$install_dir" "$profile" 2>/dev/null; then
            mkdir -p "$(dirname "$profile")"
            echo "" >> "$profile"
            echo "# Added by mnemonist installer" >> "$profile"
            echo "set -Ux fish_user_paths $install_dir \$fish_user_paths" >> "$profile"
            echo "Added $install_dir to $profile"
            echo "Restart your shell or run: source $profile"
        fi
    elif [ -n "$profile" ] && ! grep -q "$install_dir" "$profile" 2>/dev/null; then
        echo "" >> "$profile"
        echo "# Added by mnemonist installer" >> "$profile"
        echo "export PATH=\"$install_dir:\$PATH\"" >> "$profile"
        echo "Added $install_dir to $profile"
        echo "Restart your shell or run: source $profile"
    fi
}

err() {
    echo "Error: $1" >&2
    exit 1
}

main
