#!/usr/bin/env python3
"""Minimal, stdlib-only HTTP preview for neurocascade's committed results.

neurocascade has no web frontend: its "output" is the set of generated figures
(`results/*.png`) and text reports (`results/*.txt`, recursively) that the
example and report scripts write. This server turns that directory into a
browseable page so the Freebuff preview button has something meaningful to show.

It is deliberately dependency-free: it only imports `http.server`, `html`,
`json` and `pathlib`, so it works even before the scientific stack is installed.

Usage:
    python scripts/preview_server.py            # serve on $PORT (default 8080)
    python scripts/preview_server.py --check    # build the index, print a summary, exit
"""

from __future__ import annotations

import argparse
import html
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"

#: Only these suffixes under results/ are treated as browser-viewable output.
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".svg"}
TEXT_SUFFIXES = {".txt", ".md", ".csv", ".json"}

INDEX_STYLE = """
  body { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
         margin: 0; background: #0b0f14; color: #e7edf3; }
  .wrap { max-width: 1080px; margin: 0 auto; padding: 40px 28px 80px; }
  header { border-bottom: 1px solid #26313c; padding-bottom: 24px; margin-bottom: 32px; }
  h1 { font-size: 28px; margin: 0 0 8px; letter-spacing: -0.02em; }
  h2 { font-size: 16px; text-transform: uppercase; letter-spacing: 0.08em;
       color: #8fb3c8; margin: 40px 0 16px; }
  .sub { color: #93a1ad; font-size: 14px; line-height: 1.5; max-width: 72ch; }
  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
          gap: 20px; }
  figure { margin: 0; background: #111821; border: 1px solid #223041;
           border-radius: 10px; overflow: hidden; }
  figure img { width: 100%; height: 200px; object-fit: cover; display: block;
               background: #0b0f14; }
  figcaption { padding: 12px 14px; font-size: 13px; color: #c6d2dc; }
  figcaption a { color: #7fc4e8; text-decoration: none; }
  ul { list-style: none; padding: 0; margin: 0; display: grid; gap: 8px; }
  li a { color: #7fc4e8; text-decoration: none; font-size: 14px; }
  pre { background: #101720; border: 1px solid #223041; border-radius: 10px;
        padding: 16px; overflow-x: auto; font-size: 12px; line-height: 1.45;
        color: #b9c6d2; white-space: pre-wrap; }
  .note { color: #7c8b98; font-size: 12px; }
"""


def _relative_paths():
    """All viewable files under results/, relative to REPO, sorted by name."""
    found = []
    if not RESULTS.is_dir():
        return found
    for path in sorted(RESULTS.rglob("*")):
        if path.is_file() and path.suffix.lower() in (IMAGE_SUFFIXES | TEXT_SUFFIXES):
            found.append(path.relative_to(REPO))
    return found


def _title(rel: Path) -> str:
    return rel.name.replace("_", " ").replace(".png", "").replace(".txt", "")


def build_index() -> str:
    files = _relative_paths()
    figures = [p for p in files if p.suffix.lower() in IMAGE_SUFFIXES]
    reports = [p for p in files if p.suffix.lower() in TEXT_SUFFIXES]

    figure_html = ""
    for p in figures:
        figure_html += (
            f'<figure><a href="/{p.as_posix()}">'
            f'<img src="/{p.as_posix()}" alt="{html.escape(_title(p))}" loading="lazy">'
            f"</a><figcaption>{html.escape(_title(p))}</figcaption></figure>\n"
        )

    report_html = "".join(
        f'<li><a href="/{p.as_posix()}">{html.escape(p.as_posix())}</a></li>\n'
        for p in reports
    )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>neurocascade · results</title>
<style>{INDEX_STYLE}</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>neurocascade</h1>
    <p class="sub">A simulation engine modelling how a drug binding a receptor
    propagates through brain circuits to a predicted behavioural effect:
    pharmacokinetics &rarr; receptor occupancy &rarr; circuit dynamics &rarr;
    network &rarr; behaviour, as one coupled ODE system.</p>
  </header>

  <h2>Figures ({len(figures)})</h2>
  <div class="grid">{figure_html}</div>

  <h2>Reports ({len(reports)})</h2>
  <ul>{report_html}</ul>

  <p class="note">These are the committed artifacts under <code>results/</code>.
  Regenerate them by running the scripts in <code>examples/</code> and
  <code>scripts/</code>, or verify the engine with
  <code>.venv/bin/python -m pytest -m "not slow"</code>.</p>
</div>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        if self.path.split("?", 1)[0] == "/":
            body = build_index().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        rel = self.path.split("?", 1)[0].lstrip("/")
        target = (REPO / rel).resolve()
        try:
            target.relative_to(REPO.resolve())
        except ValueError:
            self.send_error(403, "Forbidden")
            return

        if not target.is_file():
            self.send_error(404, "Not found")
            return

        suffix = target.suffix.lower()
        if suffix in IMAGE_SUFFIXES:
            ctype = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
                     "gif": "image/gif", "svg": "image/svg+xml"}.get(suffix[1:], "application/octet-stream")
        elif suffix in TEXT_SUFFIXES:
            ctype = "text/plain; charset=utf-8"
        else:
            self.send_error(404, "Not found")
            return

        data = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt, *args):  # keep preview logs terse
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8080")))
    ap.add_argument("--check", action="store_true",
                    help="build the index, print a summary, and exit")
    args = ap.parse_args()

    files = _relative_paths()
    if args.check:
        print(json.dumps({
            "figures": sum(1 for p in files if p.suffix.lower() in IMAGE_SUFFIXES),
            "reports": sum(1 for p in files if p.suffix.lower() in TEXT_SUFFIXES),
            "index_bytes": len(build_index().encode("utf-8")),
        }, indent=2))
        return 0

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"neurocascade preview on http://{args.host}:{args.port} "
          f"(serving {len(files)} files under results/)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
