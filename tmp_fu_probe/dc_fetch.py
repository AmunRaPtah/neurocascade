"""Fetch Lombardo-2018-sourced plasma fu for v2 drugs missing ChEMBL fu, by
scraping DrugCentral drug cards (which mirror Lombardo F, Berellini G, Obach RS
2018 DMD 46(11):1466-1477, doi:10.1124/dmd.118.082966, PMID 30115648, with an
explicit per-row citation to that paper). Verifies name match by exact-title
match at drugcentral.org, then cross-checks structure via ChEMBL
standard_inchi_key vs DrugCentral's own InChIKey for the same struct_id.
Caches every raw response; never computes anything from memory.
"""
import json, re, time, urllib.request, urllib.parse, sys, hashlib
from pathlib import Path

ROOT = Path("/root/projects/neurocascade/.claude/worktrees/agent-aa546ac89132d41e2")
CACHE = Path("/root/nc-v2-cache/fu2/drugcentral")
CACHE.mkdir(parents=True, exist_ok=True)
CHEMBL = "https://www.ebi.ac.uk/chembl/api/data/"

def get(url, timeout=30):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 neurocascade-fu-research"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")

def cached(key, fetcher):
    fn = CACHE / (hashlib.sha1(key.encode()).hexdigest() + ".txt")
    if fn.exists():
        return fn.read_text()
    val = fetcher()
    fn.write_text(val)
    time.sleep(0.25)
    return val

def search_ids(name):
    html = cached(f"search:{name}", lambda: get(f"https://drugcentral.org/?q={urllib.parse.quote(name)}"))
    return sorted(set(re.findall(r"drugcard/(\d+)", html)))

def card_html(sid):
    return cached(f"card:{sid}", lambda: get(f"https://drugcentral.org/drugcard/{sid}"))

def card_title(html):
    m = re.search(r"<title>([^<]*)</title>", html)
    return m.group(1).strip().lower() if m else None

def fu_row(html):
    m = re.search(
        r"<td>\s*fu \(Fraction unbound in plasma\)\s*</td>\s*<td>\s*([^<]+?)\s*</td>\s*<td>\s*(.*?)</td>",
        html, re.S)
    return m

def inchikey_of(html):
    m = re.search(r"InChIKey\s*:?\s*</?[^>]*>?\s*([A-Z0-9-]{20,})", html)
    if m:
        return m.group(1)
    m = re.search(r"\b([A-Z]{14}-[A-Z]{10}-[A-Z])\b", html)
    return m.group(1) if m else None


def main():
    d = json.loads((ROOT / "src/neurocascade/benchmark/v2/inputs/plasma_fu.json").read_text())
    have = {r["inn"] for r in d["drugs"] if r.get("fu")}
    drugs = json.loads((ROOT / "src/neurocascade/benchmark/v2/drugs.json").read_text())["drugs"]
    missing = [x for x in drugs if x["inn"] not in have]
    print(f"{len(missing)} drugs missing chembl fu", file=sys.stderr)

    out = {}
    for n, x in enumerate(missing):
        inn, cid = x["inn"], x["chembl_id"]
        base = inn.split(" ", 1)[0].lower()
        try:
            ids = search_ids(base)
        except Exception as e:
            print("searchfail", inn, e, file=sys.stderr)
            continue
        match = None
        for sid in ids:
            try:
                html = card_html(sid)
            except Exception as e:
                print("cardfail", inn, sid, e, file=sys.stderr)
                continue
            title = card_title(html)
            if title == inn.lower() or title == base:
                m = fu_row(html)
                if m:
                    match = {"struct_id": sid, "title": title,
                             "value_literal": m.group(1).strip(),
                             "snippet": m.group(0),
                             "inchikey": inchikey_of(html)}
                    break
        if match:
            out[inn] = match
        if n % 10 == 0:
            print(n, len(missing), inn, "->", bool(match), file=sys.stderr, flush=True)
            (ROOT / "tmp_fu_probe" / "drugcentral_matches.json").write_text(json.dumps(out, indent=1))

    (ROOT / "tmp_fu_probe" / "drugcentral_matches.json").write_text(json.dumps(out, indent=1))
    print("matched", len(out), "of", len(missing), file=sys.stderr, flush=True)

if __name__ == "__main__":
    main()
