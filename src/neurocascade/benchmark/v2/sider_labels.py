"""SIDER-powered liability labels for Phase 35 (P35.2), train split only.

Every rule here is fixed by ``docs/prereg-p35-go.md`` (sealed 2026-10-29),
except the points that document leaves open; those are pinned below and in
docs/architecture.md ("P35.2"), written before any count was looked at.

* Split: held-out iff ``int(sha256(inn), 16) % 10`` is 0 or 1. The INN is
  stripped and lower-cased first (every v2 INN already is), the whole digest is
  the integer, and nothing is salted. This is NOT ``v2_heldout`` (salted,
  first 8 hex digits, ``% 5 == 0``); the prereg fixes a new rule.
* Label: a drug is positive for a liability when at least one mapped PT has a
  placebo-adjusted ``frequency_upper`` >= 1%. Per drug x PT the drug frequency
  is the largest ``frequency_upper`` of its non-placebo rows and the placebo
  frequency the largest of its placebo rows (SIDER 4.1 does not pair a drug
  row with its own placebo arm, so max - max is the conservative reading).
  Arithmetic is on the published decimal strings, so 0.03 - 0.02 is exactly
  0.01. A PT with no placebo row is ``placebo_uncontrolled`` and its raw
  frequency is used.
* Negative: the drug has a SIDER record (a row in ``meddra_freq`` or
  ``meddra_all_se``) and no mapped PT reaches 1%. No record: ``excluded``.
* Matching a drug to SIDER rows: the STITCH flat id (``CID1`` + 8 digits) must
  be the drug's PubChem CID or share its connectivity, AND the STITCH stereo
  id (``CID0`` + 8 digits) must belong to the drug (``match_stereo``). The
  stereo condition is the one addition to the prereg's flat-id rule: STITCH
  files citalopram and escitalopram, cetirizine and levocetirizine, ketamine
  and esketamine under one flat id, and flat-only matching would hand one
  drug's label to another (house rule 2).
"""

from __future__ import annotations

import collections
import gzip
import hashlib
import json
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Mapping

SIDER_DIR = Path(__file__).resolve().parent / "sider"
CID_MAP = SIDER_DIR / "cid_map.json"
TRAIN_ROWS = SIDER_DIR / "sider_train_rows.json.gz"
TRAIN_LABELS = SIDER_DIR / "labels_train.json"
PT_LISTING = SIDER_DIR / "pt_listing.json"
UNIVERSE = SIDER_DIR / "universe_train.json"

# Verbatim from docs/prereg-p35-go.md, "MedDRA PT -> liability map (fixed now)".
LIABILITY_PTS: dict[str, tuple[str, ...]] = {
    "sedation": ("Somnolence", "Sedation", "Hypersomnia"),
    "anticholinergic": ("Dry mouth", "Constipation", "Urinary retention", "Blurred vision",
                        "Mydriasis"),
    "eps": ("Extrapyramidal disorder", "Akathisia", "Dystonia", "Parkinsonism",
            "Tardive dyskinesia", "Tremor", "Muscle rigidity"),
    "orthostatic_hypotension": ("Orthostatic hypotension", "Postural dizziness"),
    "seizure": ("Seizure", "Convulsion", "Epilepsy", "Grand mal convulsion"),
}
THRESHOLD = Decimal("0.01")

# The SIDER 4.1 release files (Kuhn et al. 2016, Nucleic Acids Res 44:D1075,
# PMID 26481350). The EMBL host timed out from this sandbox on 2026-10-29 and
# the prereg's datalake copy was not readable by the session user, so the
# bytes come from an unmodified mirror; row counts equal the prereg's
# (291,632 frequency rows) and its per-term drug counts.
SIDER_FILES = {
    "meddra_freq.tsv.gz": "fcdfbf479c6b568dfca557a3eb2cdbe49c4a3291090999c0de0bd7092feaec7f",
    "meddra_all_se.tsv.gz": "119b2f5319a9398da83e5fe3419889010dbacf8d3eef590251b00c025e2b3f99",
}
SIDER_UPSTREAM = "http://sideeffects.embl.de/media/download/"
SIDER_MIRROR = "https://raw.githubusercontent.com/dhimmel/SIDER4/master/download/"

_PT_TO_LIABILITY = {pt.casefold(): liab for liab, pts in LIABILITY_PTS.items() for pt in pts}


def p35_split_digit(inn: str) -> int:
    return int(hashlib.sha256(inn.strip().lower().encode()).hexdigest(), 16) % 10


def p35_heldout(inn: str) -> bool:
    """The Phase 35 held-out rule: sha256(inn) mod 10 in {0, 1}."""
    return p35_split_digit(inn) in (0, 1)


def liability_of(side_effect_name: str) -> str | None:
    return _PT_TO_LIABILITY.get(side_effect_name.casefold())


# ------------------------------------------------------------------ SIDER rows

def stitch_cid(stitch_id: str) -> int:
    """``CID1xxxxxxxx`` (flat) or ``CID0xxxxxxxx`` (stereo) -> PubChem CID."""
    if len(stitch_id) != 12 or not stitch_id.startswith(("CID0", "CID1")):
        raise ValueError(f"not a STITCH compound id: {stitch_id!r}")
    return int(stitch_id[4:])


def read_freq(lines: Iterable[str]) -> list[dict[str, Any]]:
    """``meddra_freq.tsv`` lines -> row dicts (column order per SIDER 4.1 README)."""
    out = []
    for line in lines:
        f = line.rstrip("\n").split("\t")
        out.append({"flat": stitch_cid(f[0]), "stereo": stitch_cid(f[1]), "placebo": f[3] == "placebo",
                    "frequency_desc": f[4], "frequency_lower": f[5], "frequency_upper": f[6],
                    "meddra_type": f[7], "side_effect_name": f[9]})
    return out


def read_all_se(lines: Iterable[str]) -> list[dict[str, Any]]:
    out = []
    for line in lines:
        f = line.rstrip("\n").split("\t")
        out.append({"flat": stitch_cid(f[0]), "stereo": stitch_cid(f[1]), "meddra_type": f[3],
                    "side_effect_name": f[5]})
    return out


def read_gz_lines(path: Path) -> list[str]:
    raw = path.read_bytes()
    want = SIDER_FILES.get(path.name)
    if want and hashlib.sha256(raw).hexdigest() != want:
        raise ValueError(f"{path} does not match the frozen SIDER sha256 {want}")
    return gzip.decompress(raw).decode().splitlines()


SALT_WORDS = ("hydrochloride", "dihydrochloride", "hcl", "chloride", "bromide", "hydrobromide",
              "maleate", "fumarate", "sulfate", "tartrate", "bitartrate", "citrate", "mesylate",
              "besylate", "succinate", "acetate", "phosphate", "oxalate", "monohydrate",
              "hemihydrate", "dihydrate")


def names_inn(synonyms: Iterable[str], inn: str) -> list[str]:
    """PubChem synonyms that are ``inn`` or ``inn`` followed by salt words."""
    out = []
    for syn in synonyms:
        words = syn.casefold().split(" ")
        if words[0] == inn.casefold() and all(w in SALT_WORDS for w in words[1:]):
            out.append(syn)
    return out


def match_stereo(inn: str, drug_cids: Iterable[int], connectivity_cids: Iterable[int],
                 flat_to_stereo: Mapping[int, Iterable[int]],
                 stereo_parents: Mapping[int, Iterable[int]],
                 stereo_synonyms: Mapping[int, Iterable[str]] | None = None,
                 other_inns: Iterable[str] = ()) -> dict[str, Any]:
    """Which SIDER flat and stereo ids belong to one drug.

    A stereo id under a matched flat id belongs to the drug when (structural)
    it, or its PubChem parent/component, is one of the drug's CIDs; or (by
    name, for salt and unspecified-stereo records PubChem does not link to
    the drug's CID) one of its PubChem synonyms is the INN or the INN plus salt
    words and none names another v2 drug that way. Enantiomer partners
    (citalopram for escitalopram, ketamine for esketamine) fail both tests.
    """
    own = set(drug_cids)
    syns = stereo_synonyms or {}
    others = [o for o in other_inns if o != inn]
    flats = sorted(f for f in flat_to_stereo if f in own | set(connectivity_cids))
    matched, by_name, rejected = set(), {}, set()
    for f in flats:
        for s in flat_to_stereo[f]:
            if s in own or own & set(stereo_parents.get(s, ())):
                matched.add(s)
                continue
            hit = names_inn(syns.get(s, ()), inn)
            if hit and not any(names_inn(syns.get(s, ()), o) for o in others):
                matched.add(s)
                by_name[s] = hit[0]
            else:
                rejected.add(s)
    return {"flat": flats, "stereo": sorted(matched),
            "stereo_by_name": {str(k): v for k, v in sorted(by_name.items())},
            "stereo_rejected": sorted(rejected - matched)}


# ---------------------------------------------------------------------- labels

def _freq(row: Mapping[str, Any]) -> Decimal:
    # frequency_upper is never empty in SIDER 4.1; the prereg's fallback to a
    # parsed frequency_desc is therefore never reached and is not implemented.
    return Decimal(row["frequency_upper"])


def label_drug(mapped_rows: Iterable[Mapping[str, Any]], has_record: bool,
               listed_pts: Iterable[str] = (), n_freq_rows: int | None = None) -> dict[str, dict[str, Any]]:
    """Per-liability label for one drug from its mapped-PT frequency rows.

    ``listed_pts``: mapped PT names the drug lists in meddra_all_se, reported
    (never used) when they carry no frequency row. ``n_freq_rows``: the drug's
    total meddra_freq row count; 0 marks a negative that rests on no frequency
    information at all (reported, never used: the prereg counts it negative).
    """
    if not has_record:
        return {liab: {"label": "excluded"} for liab in LIABILITY_PTS}
    per_pt: dict[str, dict[str, Any]] = {}
    for r in mapped_rows:
        name = r["side_effect_name"].casefold()
        if name not in _PT_TO_LIABILITY:
            raise ValueError(f"unmapped PT passed to label_drug: {r['side_effect_name']!r}")
        e = per_pt.setdefault(name, {"pt": r["side_effect_name"], "drug": None, "placebo": None,
                                     "descs": set()})
        key = "placebo" if r["placebo"] else "drug"
        e[key] = max(e[key], _freq(r)) if e[key] is not None else _freq(r)
        if not r["placebo"]:
            e["descs"].add(r["frequency_desc"])
    out: dict[str, dict[str, Any]] = {}
    listed = {p.casefold() for p in listed_pts}
    for liab, pts in LIABILITY_PTS.items():
        rows = []
        for pt in pts:
            e = per_pt.get(pt.casefold())
            if e is None or e["drug"] is None:
                continue
            adj = e["drug"] - e["placebo"] if e["placebo"] is not None else e["drug"]
            rows.append({"pt": e["pt"], "drug_frequency_upper": str(e["drug"]),
                         "placebo_frequency_upper": None if e["placebo"] is None else str(e["placebo"]),
                         "adjusted": str(adj), "placebo_uncontrolled": e["placebo"] is None,
                         "meets_threshold": adj >= THRESHOLD,
                         "frequency_desc": sorted(e["descs"])})
        hits = [r for r in rows if r["meets_threshold"]]
        with_freq = {r["pt"].casefold() for r in rows}
        out[liab] = {
            "label": 1 if hits else 0,
            "pts": rows,
            # Positive only through PTs that have no placebo row.
            "positive_only_placebo_uncontrolled": bool(hits) and all(r["placebo_uncontrolled"] for r in hits),
            # Positive only because an upper bound sits exactly on 1% (the
            # "infrequent"/"uncommon"/"<1%" bands, 0.1-1%); disclosed, not used.
            "positive_only_at_boundary": bool(hits) and all(Decimal(r["adjusted"]) == THRESHOLD for r in hits),
            "listed_without_frequency": sorted(p for p in pts if p.casefold() in listed
                                               and p.casefold() not in with_freq),
            "no_frequency_rows": n_freq_rows == 0,
        }
    return out


def build_train_labels(drug_rows: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Labels for train drugs only. ``drug_rows[inn]`` holds ``has_record``,
    ``mapped_rows`` and ``listed_pts``. A held-out INN is refused, so a held-out
    label can never be computed by this path."""
    bad = sorted(inn for inn in drug_rows if p35_heldout(inn))
    if bad:
        raise ValueError(f"P35 held-out drugs passed to the train label builder: {bad}")
    return {inn: label_drug(d["mapped_rows"], d["has_record"], d.get("listed_pts", ()),
                            d.get("n_freq_rows"))
            for inn, d in sorted(drug_rows.items())}


def label_counts(labels: Mapping[str, Mapping[str, Any]], inns: Iterable[str]) -> dict[str, dict[str, int]]:
    inns = list(inns)
    out = {}
    for liab in LIABILITY_PTS:
        rows = [labels[i][liab] for i in inns if i in labels]
        vals = [r["label"] for r in rows]
        neg = [r for r in rows if r["label"] == 0]
        out[liab] = {"positive": vals.count(1), "negative": vals.count(0),
                     "excluded": vals.count("excluded"),
                     "positive_only_placebo_uncontrolled": sum(
                         bool(r.get("positive_only_placebo_uncontrolled")) for r in rows),
                     "positive_only_at_boundary": sum(bool(r.get("positive_only_at_boundary")) for r in rows),
                     "negative_no_frequency_rows": sum(bool(r.get("no_frequency_rows")) for r in neg),
                     "negative_listed_without_frequency": sum(bool(r.get("listed_without_frequency"))
                                                              for r in neg)}
    return out


# ------------------------------------------------------------------- universe

def universe_row(inn: str, anchors: Iterable[str], binding: Mapping[str, Any] | None,
                 has_sider_record: bool, cmax_row: Mapping[str, Any] | None,
                 cmax_hidden: bool) -> dict[str, Any]:
    """Universe membership for one train drug: measured anchor Ki, SIDER
    record, Cmax value. ``cmax_hidden``: the Cmax row is stripped from this
    tree (v2 held-out), so membership is decided host-side."""
    anchor_ki = sorted(set(anchors) & set((binding or {}).get("targets", {})))
    if cmax_row is not None:
        cmax = cmax_row["status"]
    else:
        cmax = "hidden_in_session_tree" if cmax_hidden else "absent"
    if not anchor_ki or not has_sider_record or cmax in ("searched_and_empty", "absent"):
        member: bool | str = False
    elif cmax == "value":
        member = True
    else:
        member = "pending_host"
    return {"inn": inn, "anchor_ki": anchor_ki, "sider_record": has_sider_record, "cmax": cmax,
            "in_universe": member}


# ---------------------------------------------------------------------- loaders

def load_train_labels() -> dict[str, dict[str, Any]]:
    return json.loads(TRAIN_LABELS.read_text())["drugs"]


def load_universe() -> list[dict[str, Any]]:
    return json.loads(UNIVERSE.read_text())["drugs"]


def load_cid_map() -> dict[str, dict[str, Any]]:
    return json.loads(CID_MAP.read_text())["drugs"]


def load_train_rows() -> dict[str, Any]:
    return json.loads(gzip.decompress(TRAIN_ROWS.read_bytes()))


# ---------------------------------------------------------------- concordance

def concordance(sider: Mapping[str, Mapping[str, Any]], reference: Mapping[str, Mapping[str, Any]],
                liabilities: Iterable[str], bases: Iterable[str] = ("placebo_subtracted",)
                ) -> dict[str, dict[str, Any]]:
    """SIDER train labels against the v2 FDA-label truth (``truth/liabilities.json``
    rows keyed by INN), per liability, on drugs labelled 0/1 by both and whose
    FDA label has one of ``bases``. Keys of the confusion table are
    ``fda<label>_sider<label>``."""
    bases = set(bases)
    out = {}
    for liab in liabilities:
        tab = collections.Counter()
        excluded = 0
        for inn, ref in sorted(reference.items()):
            r = ref.get(liab)
            if inn not in sider or not isinstance(r, dict) or r.get("basis") not in bases \
                    or r.get("label") not in (0, 1):
                continue
            s = sider[inn][liab]["label"]
            if s == "excluded":
                excluded += 1
                continue
            tab[f"fda{r['label']}_sider{s}"] += 1
        n = sum(tab.values())
        a, b = tab["fda1_sider1"], tab["fda1_sider0"]
        c, d = tab["fda0_sider1"], tab["fda0_sider0"]
        po = (a + d) / n if n else None
        pe = ((a + b) * (a + c) + (c + d) * (b + d)) / n ** 2 if n else None
        out[liab] = {"n": n, "fda_has_label_sider_excluded": excluded,
                     "confusion": {k: tab[k] for k in ("fda1_sider1", "fda1_sider0", "fda0_sider1", "fda0_sider0")},
                     "sider_sensitivity_to_fda_positive": round(a / (a + b), 3) if a + b else None,
                     "sider_positive_rate_on_fda_negative": round(c / (c + d), 3) if c + d else None,
                     "agreement": round(po, 3) if n else None,
                     "cohen_kappa": round((po - pe) / (1 - pe), 3) if n and pe != 1 else None}
    return out
