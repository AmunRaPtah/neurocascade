"""P32.2: the ``neurocascade`` command line.

``neurocascade predict PACKAGE.(json|csv) -o report.html [--json result.json]``
runs ``predict_compound`` on a package file and writes the one-page report.
``neurocascade export-benchmark INN -o FILE.(json|csv)`` writes a benchmark
drug's preclinical package (``package_from_benchmark``: binding panel and ADME
record only, no truth) so there is a real file to run.

Exit codes: 0 report written; 2 refused for a missing core input (the missing
inputs are printed in plain English, as P27.1 asked); 1 the file is not a
valid package, or ``serve`` could not bind its interface.
``neurocascade serve`` runs the P32.3 web UI on the wg0 interface only.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

from .compound import CoreInputMissing, load_package, package_from_benchmark, predict_compound
from .package_csv import dumps_csv, load_package_csv
from .report import render_report

EXIT_OK, EXIT_INVALID, EXIT_REFUSED = 0, 1, 2


def read_package(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return load_package_csv(path)
    return load_package(path)


def json_safe(value: Any) -> Any:
    """Strict JSON: an unbounded band edge (inf) becomes null; ``bounded`` says why."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def _predict(args: argparse.Namespace) -> int:
    try:
        package = read_package(args.package)
    except (ValueError, KeyError, OSError) as exc:
        print(f"error: {args.package} is not a valid compound package: {exc}", file=sys.stderr)
        return EXIT_INVALID
    try:
        result = predict_compound(package)
    except CoreInputMissing as exc:
        print(f"refused: no prediction for {package['name']}, because a core input is missing:",
              file=sys.stderr)
        for item in exc.missing:
            print(f"  - {item}", file=sys.stderr)
        print("A missing optional input only widens a band; a missing core input cannot.",
              file=sys.stderr)
        return EXIT_REFUSED
    out = Path(args.output or Path(args.package).with_suffix(".report.html").name)
    out.write_text(render_report(result, package))
    if args.json:
        Path(args.json).write_text(json.dumps(json_safe(result), indent=1, allow_nan=False) + "\n")
    print(f"wrote {out}")
    return EXIT_OK


def _export(args: argparse.Namespace) -> int:
    try:
        package = package_from_benchmark(args.inn)
    except (KeyError, FileNotFoundError):
        print(f"error: {args.inn!r} is not a benchmark drug", file=sys.stderr)
        return EXIT_INVALID
    out = Path(args.output)
    if out.suffix.lower() == ".csv":
        out.write_text(dumps_csv(package))
    else:
        out.write_text(json.dumps(package, indent=1) + "\n")
    print(f"wrote {out}")
    return EXIT_OK


def _serve(args: argparse.Namespace) -> int:
    from .webui import serve
    try:
        serve(args.interface, args.port)
    except OSError as exc:
        print(f"error: cannot serve on interface {args.interface}: {exc}", file=sys.stderr)
        return EXIT_INVALID
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="neurocascade",
        description="Predict human consequences of a compound's preclinical package. "
                    "Not validated: P34 returned no verdict.")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("predict", help="package file (.json or .csv) -> one-page HTML report")
    p.add_argument("package")
    p.add_argument("-o", "--output", help="report path (default: <package>.report.html)")
    p.add_argument("--json", help="also write the full prediction as strict JSON")
    p.set_defaults(func=_predict)
    e = sub.add_parser("export-benchmark",
                       help="write a benchmark drug's preclinical package (no truth) to a file")
    e.add_argument("inn")
    e.add_argument("-o", "--output", required=True, help="output path, .json or .csv")
    e.set_defaults(func=_export)
    s = sub.add_parser("serve", help="web UI (upload a package, view the report), wg0 only")
    s.add_argument("--interface", default="wg0",
                   help="bind to this interface's IPv4 address and serve only its network")
    s.add_argument("--port", type=int, default=8732)
    s.set_defaults(func=_serve)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
