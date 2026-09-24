#!/usr/bin/env bash
# Cron entry for `orchestrate.py doctor --reap --quiet --alert`.
# Loads the vault session like scripts/neurocascade-digest.sh does: without it
# the Telegram token lookup returns "" and the alert goes nowhere.
set -uo pipefail
export HOME=/root
export PATH="/root/.local/bin:/root/.nvm/versions/node/v24.18.0/bin:/usr/local/bin:/usr/bin:/bin"
# shellcheck disable=SC1091
[ -f /etc/profile.d/claude-vault.sh ] && . /etc/profile.d/claude-vault.sh
exec python3 /root/projects/neurocascade/pipeline/orchestrate.py doctor --reap --quiet --alert
