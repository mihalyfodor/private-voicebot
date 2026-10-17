#!/usr/bin/env bash
# Install one of scripts/launchd/*.plist as a LaunchAgent with this checkout's paths filled in, then load it
# (docs/deploy.md). The templates carry __ROOT__ (the checkout) and __HOME__ so no machine path lives in the repo.
#   scripts/install_launchd.sh asr|backup|remarkable [--dry-run]   # --dry-run prints the rendered plist only
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
name=com.voicebot.${1:-}.plist
src=$ROOT/scripts/launchd/$name
[ -f "$src" ] || { echo "usage: install_launchd.sh asr|backup|remarkable [--dry-run]" >&2; exit 1; }
render() { sed -e "s|__ROOT__|$ROOT|g" -e "s|__HOME__|$HOME|g" "$src"; }
if [ "${2:-}" = --dry-run ]; then render; exit 0; fi
dst=$HOME/Library/LaunchAgents/$name
launchctl unload "$dst" 2> /dev/null || true
render > "$dst"
launchctl load "$dst"
echo "loaded $dst"
