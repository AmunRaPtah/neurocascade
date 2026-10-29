"""P32.2: a flat CSV encoding of the P32.1 compound package.

A discovery team's assay data usually lives in spreadsheets, so the CLI also
accepts a package as one CSV table.  The CSV is a lossless re-encoding of
the JSON schema ``neurocascade.compound_package/1``, not a simpler schema. The
ADME section keeps the benchmark's measurement-level rows (and their
``summary`` rows), because the P29.2 extractors that the P30.3 band was
calibrated on read exactly those rows (architecture.md, Phase 32 part one).
No aggregation is done here, and no summary is recomputed.

One row per item, with these columns (any other column is copied verbatim
into the ADME row it sits on):

* ``section=compound, key=name, value=<name>``
* ``section=binding, key=<gene>, value=<Ki in nM or empty>, units=nM,
  efficacy=<a;b>, ref=<r1;r2>``
* ``section=adme, key=<field>, row_type=measurement|summary`` plus the row's
  own fields (``value``, ``units``, ``species``, ``matrix``, ``relation``,
  ``standard_type``, ``description``, ``ref``, ``assay_chembl_id``, ...)
* ``section=human_exposure, key=cmax_unbound_nM, value=<nM>, units=nM,
  ref=<source>``
"""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any, Mapping

from .compound import SCHEMA, validate_package

FIXED_COLUMNS = ["section", "key", "row_type", "value", "units", "efficacy", "ref"]
INT_COLUMNS = {"n_censored", "n_documents", "n_measurements"}
FLOAT_COLUMNS = {"value", "min", "max", "ppb_percent"}
_LIST_SEP = ";"


def _cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return repr(value)
    return str(value)


def package_to_rows(package: Mapping[str, Any]) -> list[dict[str, str]]:
    rows = [{"section": "compound", "key": "name", "value": package["name"]}]
    for gene, entry in (package.get("binding") or {}).items():
        rows.append({
            "section": "binding", "key": gene, "value": _cell(entry.get("ki_nM")),
            "units": "nM",
            "efficacy": _LIST_SEP.join(entry.get("efficacy") or []),
            "ref": _LIST_SEP.join(entry.get("refs") or []),
        })
    for field, blob in ((package.get("adme") or {}).get("fields") or {}).items():
        for row_type in ("measurement", "summary"):
            for item in blob.get(f"{row_type}s" if row_type == "measurement" else "summary", []):
                row = {"section": "adme", "key": field, "row_type": row_type}
                row.update({k: _cell(v) for k, v in item.items()})
                rows.append(row)
    exposure = package.get("human_exposure")
    if exposure is not None:
        rows.append({"section": "human_exposure", "key": "cmax_unbound_nM",
                     "value": _cell(exposure["cmax_unbound_nM"]), "units": "nM",
                     "ref": exposure["source"]})
    return rows


def dumps_csv(package: Mapping[str, Any]) -> str:
    rows = package_to_rows(package)
    extra = sorted({k for r in rows for k in r} - set(FIXED_COLUMNS))
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=FIXED_COLUMNS + extra, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue()


def _parse(column: str, text: str) -> Any:
    if column in INT_COLUMNS:
        return int(float(text))
    if column in FLOAT_COLUMNS:
        return float(text)
    return text


def loads_csv(text: str) -> dict[str, Any]:
    """Parse a package CSV into the P32.1 package dict and validate it."""
    reader = csv.DictReader(io.StringIO(text))
    missing = {"section", "key"} - set(reader.fieldnames or [])
    if missing:
        raise ValueError(f"package CSV lacks column(s): {sorted(missing)}")
    package: dict[str, Any] = {"schema": SCHEMA}
    for line, row in enumerate(reader, start=2):
        section, key = (row.get("section") or "").strip(), (row.get("key") or "").strip()
        value = (row.get("value") or "").strip()
        if section == "compound" and key == "name":
            package["name"] = value
        elif section == "binding":
            refs = row.get("ref") or ""
            efficacy = [e for e in (row.get("efficacy") or "").split(_LIST_SEP) if e]
            if value and (row.get("units") or "nM") != "nM":
                raise ValueError(f"line {line}: binding Ki must be in nM")
            package.setdefault("binding", {})[key] = {
                "ki_nM": float(value) if value else None,
                "efficacy": efficacy or None,
                "refs": [r for r in refs.split(_LIST_SEP) if r],
            }
        elif section == "adme":
            row_type = row.get("row_type")
            if row_type not in ("measurement", "summary"):
                raise ValueError(f"line {line}: adme row_type must be measurement or summary")
            item = {c: _parse(c, v) for c, v in row.items()
                    if c not in ("section", "key", "row_type", "efficacy") and v not in (None, "")}
            blob = package.setdefault("adme", {"fields": {}})["fields"].setdefault(
                key, {"measurements": [], "summary": []})
            blob["measurements" if row_type == "measurement" else "summary"].append(item)
        elif section == "human_exposure" and key == "cmax_unbound_nM":
            if (row.get("units") or "nM") != "nM":
                raise ValueError(f"line {line}: human_exposure must be in nM")
            package["human_exposure"] = {"cmax_unbound_nM": float(value),
                                         "source": row.get("ref") or ""}
        else:
            raise ValueError(f"line {line}: unknown row section={section!r} key={key!r}")
    validate_package(package)
    return package


def load_package_csv(path: str | Path) -> dict[str, Any]:
    return loads_csv(Path(path).read_text())


__all__ = ["dumps_csv", "load_package_csv", "loads_csv", "package_to_rows"]
