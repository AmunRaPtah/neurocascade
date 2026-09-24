#!/usr/bin/env bash
# Cron entry for `conductor.py tick` (every 20 min). Same environment as
# doctor-cron.sh: the vault session for Telegram, node for gh/modal, and no
# stale GITHUB_TOKEN shadowing gh's stored login.
set -uo pipefail
export HOME=/root
export PATH="/root/.local/bin:/root/.nvm/versions/node/v24.18.0/bin:/usr/local/bin:/usr/bin:/bin"
# shellcheck disable=SC1091
[ -f /etc/profile.d/claude-vault.sh ] && . /etc/profile.d/claude-vault.sh
unset GITHUB_TOKEN GH_TOKEN
exec python3 /root/projects/neurocascade/pipeline/conductor.py tick
