#!/usr/bin/env python3
"""
Batched daily digest for the neurocascade autonomous build.

Reads logs/events.jsonl since the last digest and sends one Telegram summary,
rather than a message per task. Silent when nothing happened.

Usage:
  digest.py             send the digest for everything since the last run
  digest.py --print     print it without sending
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EVENTS = REPO / "logs" / "events.jsonl"
STATE = REPO / "logs" / ".digest-state.json"
BACKLOG = REPO / "tasks" / "backlog.json"
OWNER_ID = os.environ.get("TELEGRAM_OWNER_ID", "7003237201")


def bot_token() -> str:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    if token:
        return token
    try:
        r = subprocess.run(["bw-get", "Telegram - Pardalos Bot"],
                           capture_output=True, text=True, timeout=60)
        return r.stdout.strip()
    except Exception:
        return ""


def read_events(since: str) -> list[dict]:
    if not EVENTS.exists():
        return []
    out = []
    for line in EVENTS.read_text().splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not since or rec.get("ts", "") > since:
            out.append(rec)
    return out


def build_digest(events: list[dict]) -> str:
    backlog = json.loads(BACKLOG.read_text())
    tasks = backlog["tasks"]
    done = sum(1 for t in tasks if t["status"] == "done")
    total = len(tasks)

    finished = [e for e in events if e["kind"] == "finished"]
    quarantined = [e for e in events if e["kind"] == "quarantined"]
    skipped = [e for e in events if e["kind"] == "skipped"]
    launch_failed = [e for e in events if e["kind"] == "agent_launch_failed"]
    aborted = [e for e in events if e["kind"] == "aborted"]

    # A build that is stuck emits no finished events at all, so keying the
    # digest on those alone means the one day it matters is the one day nothing
    # is sent. Faults are news too.
    if not finished and not quarantined and not launch_failed and not aborted:
        return ""

    lines = [f"neurocascade autonomous build: {done}/{total} tasks done", ""]

    for e in finished:
        status = e.get("status", "?")
        mark = "done" if status == "done" else status
        line = f"  {e['task']}: {mark}"
        if e.get("commits"):
            line += f" ({e['commits']} commits)"
        if e.get("note"):
            line += f"\n     {e['note']}"
        lines.append(line)

    for e in quarantined:
        lines.append(f"  {e['task']}: FAILED test gate, work parked on {e['branch']}")

    if launch_failed:
        lines.append("")
        lines.append(f"  BROKEN: the agent could not start on {len(launch_failed)} tick(s).")
        lines.append(f"     {launch_failed[-1].get('detail', 'no detail')}")
        lines.append("     Those tasks are back to pending; nothing was built.")

    if aborted:
        reasons = {}
        for e in aborted:
            reasons[e.get("reason", "?")] = reasons.get(e.get("reason", "?"), 0) + 1
        lines.append("")
        lines.append("  ticks aborted: " +
                     ", ".join(f"{k} x{v}" for k, v in reasons.items()))

    if skipped:
        reasons = {}
        for e in skipped:
            r = e.get("reason", "?").split(":")[0]
            reasons[r] = reasons.get(r, 0) + 1
        lines.append("")
        lines.append("  ticks skipped: " +
                     ", ".join(f"{k} x{v}" for k, v in reasons.items()))

    nxt = next((t for t in tasks if t["status"] == "pending"), None)
    if nxt:
        lines.append("")
        lines.append(f"next: {nxt['id']} {nxt['title']}")

    failed = [t for t in tasks if t["status"] in ("failed", "blocked")]
    if failed:
        lines.append("")
        lines.append("needs a look: " + ", ".join(t["id"] for t in failed))

    return "\n".join(lines)


def send(text: str) -> bool:
    token = bot_token()
    if not token:
        print("no telegram token available")
        return False
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = urllib.parse.urlencode({"chat_id": OWNER_ID, "text": text}).encode()
    try:
        with urllib.request.urlopen(url, data=data, timeout=20) as r:
            return r.status < 300
    except Exception as e:
        print(f"telegram send failed: {e}")
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--print", dest="only_print", action="store_true")
    args = ap.parse_args()

    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    since = state.get("last_ts", "")

    events = read_events(since)
    text = build_digest(events)

    if not text:
        print("nothing to report")
        return 0

    if args.only_print:
        print(text)
        return 0

    if send(text) and events:
        STATE.parent.mkdir(exist_ok=True)
        STATE.write_text(json.dumps(
            {"last_ts": max(e["ts"] for e in events),
             "sent_at": datetime.now(timezone.utc).isoformat()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
