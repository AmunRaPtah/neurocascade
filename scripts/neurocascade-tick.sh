#!/usr/bin/env bash
# Cron entry point for the neurocascade autonomous build.
#
# flock guarantees a single tick at a time: a task can run for over two hours,
# and cron fires more often than that on purpose so the next task starts
# promptly once one finishes. Overlapping ticks exit immediately.
set -uo pipefail

export HOME=/root
export PATH="/root/.nvm/versions/node/v24.18.0/bin:/usr/local/bin:/usr/bin:/bin"

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
