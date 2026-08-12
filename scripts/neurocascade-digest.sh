#!/usr/bin/env bash
# Cron entry point for the neurocascade digest.
#
# Wrapped rather than called directly from cron so the vault session is loaded:
# the Telegram bot token comes from `bw-get`, which reads BW_SESSION out of
# /etc/profile.d/claude-vault.sh, and that script needs HOME set before it is
# sourced. Called directly from cron the token lookup silently returns "", the
# digest logs "no telegram token available", and the build reports to nobody.
# That is what happened every day until 2026-08-12.
set -uo pipefail

export HOME=/root
export PATH="/root/.nvm/versions/node/v24.18.0/bin:/usr/local/bin:/usr/bin:/bin"

# shellcheck disable=SC1091
[ -f /etc/profile.d/claude-vault.sh ] && . /etc/profile.d/claude-vault.sh

REPO=/root/projects/neurocascade

if [[ " $* " == *" --print "* ]]; then
  exec /root/projects/pardalos/.venv/bin/python "$REPO/scripts/digest.py" "$@"
fi

exec /root/projects/pardalos/.venv/bin/python "$REPO/scripts/digest.py" "$@" \
  >> "$REPO/logs/digest.log" 2>&1
