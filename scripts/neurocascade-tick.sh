#!/usr/bin/env bash
# Cron entry point for the neurocascade autonomous build.
#
# flock guarantees a single tick at a time: a task can run for over two hours,
# and cron fires more often than that on purpose so the next task starts
# promptly once one finishes. Overlapping ticks exit immediately.
set -uo pipefail

export HOME=/root
# /usr/local/bin MUST precede the nvm bin dir. `claude` exists in both: the nvm
# one is the raw CLI, and /usr/local/bin/claude is the wrapper that injects the
# pinned durable setup token from /root/.claude-anthropic/oauth-token. With nvm
# first, the runner got the raw binary, which falls back to
# ~/.claude/.credentials.json, whose refresh token expired 2026-08-18, and every
# tick died with "Failed to authenticate: OAuth session expired and could not be
# refreshed". The ordering was harmless only while ticks were routed to
# OpenRouter, which masked it. The wrapper resolves the newest nvm binary itself,
# so the nvm entry is still needed for node, just not first.
export PATH="/usr/local/bin:/root/.nvm/versions/node/v24.18.0/bin:/usr/bin:/bin"

# The runner drives `claude -p --permission-mode bypassPermissions`, which the
# CLI refuses to start as root unless IS_SANDBOX is set. An interactive session
# exports it, so ticks launched by hand inherited it and worked; cron starts
# from an empty environment, so every cron tick failed at agent launch and did
# nothing. Set it explicitly rather than depending on who started the tick.
export IS_SANDBOX=1

# BW_SESSION and any vault-backed credentials.
# shellcheck disable=SC1091
[ -f /etc/profile.d/claude-vault.sh ] && . /etc/profile.d/claude-vault.sh

REPO=/root/projects/neurocascade
LOG="$REPO/logs/tick.log"
mkdir -p "$REPO/logs"

exec /usr/bin/flock -n /tmp/neurocascade-autobuild.lock \
  /usr/bin/nice -n 10 /usr/bin/ionice -c3 \
  /root/projects/pardalos/.venv/bin/python "$REPO/scripts/autonomous_runner.py" \
  >> "$LOG" 2>&1
