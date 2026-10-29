"""P32.3: the wg0-only web UI serves predict_compound with the label on every row."""

from __future__ import annotations

import http.client
import ipaddress
import json
import re
import threading
import uuid

import pytest

from neurocascade import webui
from neurocascade.cli import main
from neurocascade.compound import PROVENANCE_LABEL, SCHEMA, package_from_benchmark
from neurocascade.package_csv import dumps_csv
from neurocascade.report import audit_report

CLEAN = {"unlinked_numbers": [], "rows_without_label": [], "broken_links": []}
LABEL = f'<span class="nv">{PROVENANCE_LABEL}</span>'


@pytest.fixture(scope="module")
def server():
    srv = webui.make_server(ipaddress.IPv4Interface("127.0.0.1/8"), 0)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield srv
    srv.shutdown()
    srv.server_close()


def _request(srv, method, path, body=None, headers=None):
    conn = http.client.HTTPConnection(*srv.server_address[:2], timeout=120)
    conn.request(method, path, body=body, headers=headers or {})
    resp = conn.getresponse()
    out = resp.status, resp.read().decode()
    conn.close()
    return out


def _upload(srv, data: bytes, filename: str):
    boundary = uuid.uuid4().hex
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"package\"; "
            f"filename=\"{filename}\"\r\nContent-Type: application/octet-stream\r\n\r\n"
            ).encode() + data + f"\r\n--{boundary}--\r\n".encode()
    return _request(srv, "POST", "/predict", body,
                    {"Content-Type": f"multipart/form-data; boundary={boundary}"})


def test_index_has_upload_form_and_label(server):
    status, text = _request(server, "GET", "/")
    assert status == 200
    assert 'type="file" name="package"' in text and LABEL in text


@pytest.mark.parametrize("fmt", ["csv", "json"])
def test_upload_returns_audited_report_with_label_on_every_row(server, fmt):
    package = package_from_benchmark("haloperidol")
    data = dumps_csv(package) if fmt == "csv" else json.dumps(package)
    status, text = _upload(server, data.encode(), f"haloperidol.{fmt}")
    assert status == 200
    assert audit_report(text) == CLEAN
    rows = re.findall(r'<tr class="quantity[^"]*">(.*?)</tr>', text, flags=re.S)
    assert len(rows) == 10 and all(LABEL in r for r in rows)


def test_api_predict_returns_strict_json(server):
    status, text = _request(server, "POST", "/api/predict",
                            json.dumps(package_from_benchmark("haloperidol")),
                            {"Content-Type": "application/json"})
    assert status == 200
    result = json.loads(text)
    assert result["compound"] == "haloperidol"
    assert result["human_pk"]["clearance"]["provenance"]["label"] == PROVENANCE_LABEL


def test_refusal_and_invalid_package(server):
    refused = json.dumps({"schema": SCHEMA, "name": "nothing", "binding": {}}).encode()
    status, text = _upload(server, refused, "nothing.json")
    assert status == 422 and "binding panel" in text
    status, text = _request(server, "POST", "/api/predict", refused)
    assert status == 422 and json.loads(text)["missing"]
    status, _ = _upload(server, b"{not json", "x.json")
    assert status == 400
    status, _ = _request(server, "GET", "/nope")
    assert status == 404


def test_demo_page_passes_audit(server):
    status, text = _request(server, "GET", "/demo")
    assert status == 200
    assert audit_report(text) == CLEAN
    assert text.count(LABEL) >= 5 * 8


def test_clients_outside_the_wg_network_are_dropped():
    srv = webui.make_server(ipaddress.IPv4Interface("127.0.0.1/8"), 0)
    srv.allowed = ipaddress.IPv4Network("10.99.0.0/24")
    assert srv.verify_request(None, ("10.99.0.7", 5000)) is True
    assert srv.verify_request(None, ("127.0.0.1", 5000)) is False
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        with pytest.raises((http.client.RemoteDisconnected, ConnectionError)):
            _request(srv, "GET", "/")
    finally:
        srv.shutdown()
        srv.server_close()


def test_never_binds_all_interfaces():
    with pytest.raises(ValueError, match="wg0-only"):
        webui.make_server(ipaddress.IPv4Interface("0.0.0.0/0"), 0)


def test_interface_lookup():
    assert webui.interface_network("lo") == ipaddress.IPv4Interface("127.0.0.1/8")
    with pytest.raises(OSError):
        webui.interface_network("nc-absent0")
    assert main(["serve", "--interface", "nc-absent0"]) == 1
