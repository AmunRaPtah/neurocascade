# Sandbox for the open-weights worker agent. See pipeline/README.md.
#
# The worker gets this image, its own throwaway clone at /work, data/ mounted
# read-only, and a network path to the LLM proxy only. It never sees /root,
# never holds a GitHub credential, and runs as an unprivileged user, so a
# destructive command can at worst wreck its own clone.
#
#   docker build -f pipeline/worker.Dockerfile -t nc-worker .
FROM node:24-bookworm-slim AS node

FROM python:3.12-slim-bookworm
COPY --from=node /usr/local/bin/node /usr/local/bin/node
COPY --from=node /usr/local/lib/node_modules /usr/local/lib/node_modules
RUN ln -s /usr/local/lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
 && apt-get update && apt-get install -y --no-install-recommends git ripgrep ca-certificates \
 && rm -rf /var/lib/apt/lists/*

ARG CLAUDE_CODE_VERSION=2.1.281
RUN npm install -g "@anthropic-ai/claude-code@${CLAUDE_CODE_VERSION}" && npm cache clean --force

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

RUN useradd -m -u 20001 worker \
 && git config --system user.name "nc-worker" \
 && git config --system user.email "nc-worker@localhost" \
 && git config --system --add safe.directory /work
USER worker
WORKDIR /work
ENV DISABLE_AUTOUPDATER=1 DISABLE_TELEMETRY=1 DISABLE_ERROR_REPORTING=1 \
    CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 PYTHONDONTWRITEBYTECODE=1
