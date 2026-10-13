"""Host-side guards on architect patches (2026-10-12 audit).

Three things the architect must not decide for itself:

  held-out   the P28.1 frozen split. Held-out drugs' truth is removed from the
             trees architect and reviewer sessions see, and a patch may not
             touch split.json or any file held-out truth lives in.
  citations  every PMID a patch adds to benchmark data must exist in PubMed and
             name the record's drug in its title, abstract or MeSH substances,
             unless the record says why not in a "pmid_note".
  n          a patch that adds a validation-ledger row must ship
             results/<task>.json with "n_scored"; n_scored == 0 is EMPTY, not
             COMPLETE, whatever the session claimed.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

BENCH = "src/neurocascade/benchmark/"
SPLIT = BENCH + "split.json"
TRUTH = BENCH + "truth/"
#: Truth files holding held-out rows next to (or instead of) train rows. Their
#: held-out rows are filtered out of session trees, and a patch may not touch
#: them at all, since its context would not match the full file anyway.
TRUTH_AGGREGATES = (TRUTH + "effect_labels.json", BENCH + "p29_5/oral_exposure.json",
                    BENCH + "p29_5/pet_plasma_pairs.json")
LEDGER = "docs/validation-ledger.md"
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"


# ------------------------------------------------------------------- held-out

def heldout(root: Path) -> list[str]:
    """Held-out INNs of the tree at root ([] on a branch older than P28.1)."""
    f = root / SPLIT
    return json.loads(f.read_text())["heldout"] if f.exists() else []


def truth_file(inn: str) -> str:
    return f"{TRUTH}{inn.replace(' ', '_')}.json"


def frozen_paths(root: Path) -> list[str]:
    """What an architect patch may never touch: the split, every held-out
    drug's truth record, and the aggregates held-out rows are filtered from.
    New truth files for train drugs stay allowed."""
    return [SPLIT, *TRUTH_AGGREGATES, *(truth_file(inn) for inn in heldout(root))]


def hide_heldout(tree: Path, wip: str = "") -> tuple[str, bool]:
    """Strip held-out truth from a session tree and restart its git history,
    since every earlier commit still holds the full files. Then apply an
    earlier session's unfinished patch, if any. Returns the new base commit to
    diff the session's work against, and whether the wip patch applied."""
    # This deletes .git, so it must only ever see a session copy (session_tree
    # names them <task>-tree), never the real repo or a host clone.
    if not tree.name.endswith("-tree"):
        raise ValueError(f"hide_heldout refuses non-session tree {tree}")
    names = set(heldout(tree))
    for inn in names:
        (tree / truth_file(inn)).unlink(missing_ok=True)
    for rel in TRUTH_AGGREGATES:
        p = tree / rel
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        rows = d.get("drugs")
        if isinstance(rows, dict):
            d["drugs"] = {k: v for k, v in rows.items() if k not in names}
        elif isinstance(rows, list):
            d["drugs"] = [r for r in rows if r.get("inn") not in names and r.get("split") != "heldout"]
        p.write_text(json.dumps(d, indent=1, ensure_ascii=False) + "\n")
    shutil.rmtree(tree / ".git")
    # The tree is already owned by the worker uid; the host runs git as root.
    git = ["git", "-C", str(tree), "-c", "safe.directory=*", "-c", "user.name=conductor",
           "-c", "user.email=conductor@localhost", "-c", "core.hooksPath=/dev/null"]
    # data/ is gitignored; naming it in a pathspec makes `git add` exit 1.
    for args in (["init", "-q"], ["add", "-A"],
                 ["commit", "-q", "-m", "base, held-out truth removed (conductor)"]):
        subprocess.run(git + args, check=True, capture_output=True)
    base = subprocess.run(git + ["rev-parse", "HEAD"], check=True, capture_output=True,
                          text=True).stdout.strip()
    applied = bool(wip) and subprocess.run(git + ["apply", "-"], input=wip, text=True,
                                           capture_output=True).returncode == 0
    return base, applied


# ---------------------------------------------------------------------------- n

def ledger_rows_added(patch: str) -> bool:
    """Does the patch add a table row (not a separator) to the ledger?"""
    cur = None
    for line in patch.splitlines():
        if line.startswith("diff --git "):
            cur = line.split(" b/", 1)[-1]
        elif cur == LEDGER and line.startswith("+| ") and not line.startswith("+|---"):
            return True
    return False


def host_n(host: Path, tid: str, patch: str) -> tuple[int | None, str | None]:
    """(n_scored, None), (None, None) when no ledger row is added, or
    (None, why) when the results file is missing or malformed."""
    if not ledger_rows_added(patch):
        return None, None
    f = host / "results" / f"{tid}.json"
    try:
        n = json.loads(f.read_text())["n_scored"]
    except (OSError, ValueError, KeyError, TypeError):
        return None, f"a patch that adds a validation-ledger row must ship results/{tid}.json with an integer \"n_scored\""
    if not isinstance(n, int) or isinstance(n, bool) or n < 0:
        return None, f"results/{tid}.json: n_scored must be a non-negative integer, got {n!r}"
    return n, None


# -------------------------------------------------------------------- citations

class CitationsUnavailable(Exception):
    """PubMed could not be reached: unverified, not wrong."""


def pmid_refs(obj, inn: str | None = None, note: bool = False) -> list[tuple[str, str | None, bool]]:
    """(pmid, drug INN in scope, pmid_note in scope) for every non-null "pmid"."""
    out = []
    if isinstance(obj, dict):
        inn = obj.get("inn") if isinstance(obj.get("inn"), str) else inn
        note = note or bool(obj.get("pmid_note"))
        if obj.get("pmid") not in (None, ""):
            out.append((str(obj["pmid"]).strip(), inn, note))
        for v in obj.values():
            out += pmid_refs(v, inn, note)
    elif isinstance(obj, list):
        for v in obj:
            out += pmid_refs(v, inn, note)
    return out


def file_refs(path: str, text: str) -> list[tuple[str, str | None, bool]]:
    """Refs in one benchmark JSON file; a per-drug file's INN is its name."""
    try:
        obj = json.loads(text)
    except ValueError:
        return []
    p = Path(path)
    per_drug = p.parent.name in ("binding", "adme", "truth") and p.stem not in ("coverage", "effect_labels")
    return pmid_refs(obj, p.stem.replace("_", " ") if per_drug else None)


def _get(url: str, timeout: float) -> bytes:
    try:
        return urllib.request.urlopen(url, timeout=timeout).read()
    except Exception as e:
        raise CitationsUnavailable(f"{url[:80]}: {e!r}"[:200]) from e


def pubmed(pmids: list[str], cache_path: Path, fetch=_get, batch: int = 200,
           timeout: float = 30) -> dict[str, dict]:
    """{pmid: {"exists", "title", "journal", "text"}}, text being title +
    abstract + MeSH substance names. Cached per PMID in cache_path."""
    try:
        cache = json.loads(cache_path.read_text())
    except (OSError, ValueError):
        cache = {}
    pmids = [norm(p) for p in pmids]
    todo = sorted({p for p in pmids if p.isdigit()} - set(cache))
    for i in range(0, len(todo), batch):
        ids = todo[i:i + batch]
        q = urllib.parse.urlencode({"db": "pubmed", "id": ",".join(ids), "retmode": "json"})
        res = json.loads(fetch(f"{EUTILS}esummary.fcgi?{q}", timeout)).get("result", {})
        q = urllib.parse.urlencode({"db": "pubmed", "id": ",".join(ids), "retmode": "xml"})
        texts = {}
        for art in ET.fromstring(fetch(f"{EUTILS}efetch.fcgi?{q}", timeout)).iter("PubmedArticle"):
            texts[art.findtext(".//PMID")] = " ".join(
                "".join(e.itertext()) for e in art.iter()
                if e.tag in ("ArticleTitle", "AbstractText", "NameOfSubstance", "DescriptorName"))
        for p in ids:
            r = res.get(p) or {}
            ok = bool(r) and "error" not in r and r.get("uid") == p
            cache[p] = {"exists": ok, "title": r.get("title", ""),
                        "journal": r.get("fulljournalname", ""), "text": texts.get(p, "")}
        time.sleep(0.4)        # NCBI allows 3 requests/s without a key
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, indent=0))
    return {p: cache.get(p, {"exists": False, "title": "", "journal": "", "text": ""}) for p in pmids}


def norm(pmid: str) -> str:
    """PubMed ignores leading zeros (PDSP-sourced binding rows carry them)."""
    return pmid.strip().lstrip("0") or pmid


def names_drug(rec: dict, inn: str) -> bool:
    hay = re.sub(r"[\s_-]+", " ", (rec["title"] + " " + rec["text"]).lower())
    return re.sub(r"[\s_-]+", " ", inn.lower()) in hay


def check_refs(refs: list[tuple[str, str | None, bool]], cache_path: Path,
               fetch=_get) -> list[str]:
    """One line per bad citation: missing from PubMed, or not about its drug."""
    if not refs:
        return []
    recs = pubmed([p for p, _, _ in refs], cache_path, fetch)
    bad = []
    for p, inn, note in dict.fromkeys(refs):
        rec = recs[norm(p)]
        if not p.isdigit() or not rec["exists"]:
            bad.append(f"PMID {p} ({inn or 'no drug'}): not in PubMed")
        elif inn and not note and not names_drug(rec, inn):
            bad.append(f"PMID {p} ({inn}): drug not named in title/abstract/MeSH: "
                       f"{rec['title'][:90]!r} ({rec['journal'][:40]})")
    return bad


def new_refs(patch_paths: list[str], old: dict[str, str], new: dict[str, str]) -> list[tuple[str, str | None, bool]]:
    """Refs the patch introduces: in a benchmark JSON file's new text and not
    already cited for the same drug in its old text."""
    out = []
    for path in patch_paths:
        if not (path.startswith(BENCH) and path.endswith(".json")) or path not in new:
            continue
        before = {(p, i) for p, i, _ in file_refs(path, old.get(path, ""))}
        out += [r for r in file_refs(path, new[path]) if (r[0], r[1]) not in before]
    return out
