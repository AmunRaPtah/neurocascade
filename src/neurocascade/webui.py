"""P32.3: a wg0-only web UI around ``predict_compound``.

``neurocascade serve`` binds to the IPv4 address of the WireGuard interface
(``wg0`` by default), never to ``0.0.0.0``. It also answers only clients
inside that interface's network. Binding to an address is not enough on
Linux, which accepts packets for any local address on any interface, so the
check is ``verify_request``. The module uses the standard library only, so the
installed wheel (numpy and scipy) still runs it.

Routes:

* ``GET /``: an upload form for a package (.json or .csv).
* ``POST /predict``: the multipart upload. The response is the P32.2 one-page
  report. ``report.audit_report`` checks that report, and every row carries
  ``PROVENANCE_LABEL``.
* ``POST /api/predict``: a JSON package in, strict JSON prediction out.
* ``GET /demo``: the P32.3 blind-demo table on held-out benchmark drugs.

A refusal for a missing core input returns 422 and names each missing input.
A malformed package returns 400.
"""

from __future__ import annotations

import email.parser
import email.policy
import fcntl
import html
import ipaddress
import json
import socket
import struct
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .compound import PROVENANCE_LABEL, CoreInputMissing, predict_compound, validate_package
from .package_csv import loads_csv
from .report import render_report

WG_INTERFACE = "wg0"
DEFAULT_PORT = 8732
MAX_UPLOAD_BYTES = 2_000_000
_SIOCGIFADDR, _SIOCGIFNETMASK = 0x8915, 0x891B


def interface_network(name: str = WG_INTERFACE) -> ipaddress.IPv4Interface:
    """IPv4 address and netmask of a network interface (Linux ioctl).

    Raises ``OSError`` when the interface does not exist or has no IPv4 address.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        req = struct.pack("256s", name.encode()[:15])
        addr = socket.inet_ntoa(fcntl.ioctl(s.fileno(), _SIOCGIFADDR, req)[20:24])
        mask = socket.inet_ntoa(fcntl.ioctl(s.fileno(), _SIOCGIFNETMASK, req)[20:24])
    return ipaddress.IPv4Interface(f"{addr}/{mask}")


def parse_package(data: bytes, filename: str = "") -> dict[str, Any]:
    """A package from uploaded bytes: CSV by suffix, otherwise JSON."""
    text = data.decode("utf-8-sig")
    if filename.lower().endswith(".csv"):
        return loads_csv(text)
    package = json.loads(text)
    if not isinstance(package, dict):
        raise ValueError("a JSON package must be an object")
    validate_package(package)
    return package


def _multipart_file(content_type: str, body: bytes) -> tuple[bytes, str]:
    msg = email.parser.BytesParser(policy=email.policy.HTTP).parsebytes(
        b"Content-Type: " + content_type.encode() + b"\r\n\r\n" + body)
    for part in msg.iter_parts() if msg.is_multipart() else []:
        if part.get_param("name", header="content-disposition") == "package":
            return part.get_payload(decode=True) or b"", part.get_filename() or ""
    raise ValueError("upload has no 'package' file field")


def _page(title: str, body: str) -> str:
    esc = html.escape
    return (f'<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><title>{esc(title)}'
            "</title><style>body{font:14px system-ui,sans-serif;max-width:760px;margin:40px auto}"
            ".nv{color:#a40000;font-weight:600}</style></head><body>"
            f'<h1>{esc(title)}</h1><p><span class="nv">{esc(PROVENANCE_LABEL)}</span></p>'
            f"{body}</body></html>")


INDEX = _page("neurocascade: predict a compound", """
<form method="post" action="/predict" enctype="multipart/form-data">
<p>Preclinical package (<code>neurocascade.compound_package/1</code>, .json or .csv):</p>
<p><input type="file" name="package" accept=".json,.csv" required></p>
<p><button type="submit">Predict</button></p></form>
<p><a href="/demo">Blind demo on held-out benchmark drugs</a></p>
<p>Write a benchmark drug's package with <code>neurocascade export-benchmark INN -o x.csv</code>.</p>
""")


class Handler(BaseHTTPRequestHandler):
    server_version = "neurocascade-webui"
    demo_html: str | None = None

    def _send(self, status: int, body: str, ctype: str = "text/html; charset=utf-8") -> None:
        data = body.encode()
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _body(self) -> bytes | None:
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_UPLOAD_BYTES:
            self._send(413, _page("Package too large", f"<p>Limit {MAX_UPLOAD_BYTES} bytes.</p>"))
            return None
        return self.rfile.read(length)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/":
            self._send(200, INDEX)
        elif self.path == "/demo":
            if Handler.demo_html is None:
                from .blind_demo import render_html, run_blind_demo
                Handler.demo_html = render_html(run_blind_demo())
            self._send(200, Handler.demo_html)
        else:
            self._send(404, _page("Not found", ""))

    def do_POST(self) -> None:  # noqa: N802
        body = self._body()
        if body is None:
            return
        api = self.path == "/api/predict"
        if self.path not in ("/predict", "/api/predict"):
            self._send(404, _page("Not found", ""))
            return
        try:
            if api:
                package = parse_package(body, "package.json")
            else:
                data, filename = _multipart_file(self.headers.get("Content-Type", ""), body)
                package = parse_package(data, filename)
        except (ValueError, KeyError, UnicodeDecodeError) as exc:
            msg = f"not a valid compound package: {exc}"
            if api:
                self._send(400, json.dumps({"error": msg}), "application/json")
            else:
                self._send(400, _page("Invalid package", f"<p>{html.escape(msg)}</p>"))
            return
        try:
            result = predict_compound(package)
        except CoreInputMissing as exc:
            if api:
                self._send(422, json.dumps({"refused": package["name"], "missing": exc.missing,
                                            "label": PROVENANCE_LABEL}), "application/json")
            else:
                items = "".join(f"<li>{html.escape(m)}</li>" for m in exc.missing)
                self._send(422, _page(f"Refused: {package['name']}",
                                      f"<p>A core input is missing:</p><ul>{items}</ul>"))
            return
        if api:
            from .cli import json_safe
            self._send(200, json.dumps(json_safe(result), allow_nan=False), "application/json")
        else:
            self._send(200, render_report(result, package))

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        pass


class WgOnlyServer(ThreadingHTTPServer):
    """Serves only clients inside ``allowed`` (the wg0 network)."""

    daemon_threads = True

    def __init__(self, address: tuple[str, int], allowed: ipaddress.IPv4Network):
        self.allowed = allowed
        super().__init__(address, Handler)

    def verify_request(self, request: Any, client_address: tuple[str, int]) -> bool:
        return ipaddress.ip_address(client_address[0]) in self.allowed


def make_server(interface: ipaddress.IPv4Interface, port: int = DEFAULT_PORT) -> WgOnlyServer:
    """Server bound to ``interface.ip`` that accepts only ``interface.network``."""
    if interface.ip.is_unspecified:
        raise ValueError("refusing to bind 0.0.0.0: the UI is wg0-only")
    return WgOnlyServer((str(interface.ip), port), interface.network)


def serve(interface: str = WG_INTERFACE, port: int = DEFAULT_PORT) -> None:
    server = make_server(interface_network(interface), port)
    host, bound = server.server_address[:2]
    print(f"neurocascade UI on http://{host}:{bound}/ ({interface} only). {PROVENANCE_LABEL}")
    try:
        server.serve_forever()
    finally:
        server.server_close()


__all__ = ["DEFAULT_PORT", "WG_INTERFACE", "Handler", "WgOnlyServer", "interface_network",
           "make_server", "parse_package", "serve"]
