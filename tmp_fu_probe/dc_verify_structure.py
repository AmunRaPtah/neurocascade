"""Structure cross-check: for each DrugCentral fu match, fetch DrugCentral's
own InChI for the matched struct_id, and ChEMBL's canonical_smiles/standard_inchi
for the same chembl_id, and compare on the InChIKey skeleton layer (first block
of the InChIKey, which encodes connectivity) using a dependency-free InChIKey
derivation is not available without RDKit, so instead compare the full InChI
strings after normalising whitespace -- an exact match is strong confirmation,
and any mismatch drops the candidate (never used), rather than guessing from a
partial comparison.
"""
import hashlib, json, re, sys, time, urllib.request
from pathlib import Path

ROOT = Path("/root/projects/neurocascade/.claude/worktrees/agent-aa546ac89132d41e2")
CACHE = Path("/root/nc-v2-cache/fu2/drugcentral")
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


def dc_inchi(struct_id):
    html = cached(f"inchi:{struct_id}", lambda: get(f"https://drugcentral.org/drug/{struct_id}/inchi"))
    m = re.search(r"(InChI=\S+)", html)
    return m.group(1).strip() if m else None


def chembl_inchi(chembl_id):
    js = cached(f"chemblmol:{chembl_id}",
                lambda: get(f"{CHEMBL}molecule/{chembl_id}.json"))
    rec = json.loads(js)
    return (rec.get("molecule_structures") or {}).get("standard_inchi")


def main():
    matches = json.loads((ROOT / "tmp_fu_probe" / "drugcentral_matches.json").read_text())
    drugs = {d["inn"]: d["chembl_id"]
             for d in json.loads((ROOT / "src/neurocascade/benchmark/v2/drugs.json").read_text())["drugs"]}
    verified = {}
    rejected = []
    for inn, m in matches.items():
        cid = drugs[inn]
        try:
            dci = dc_inchi(m["struct_id"])
        except Exception as e:
            print("dc_inchi_fail", inn, e, file=sys.stderr)
            continue
        try:
            cmi = chembl_inchi(cid)
        except Exception as e:
            print("chembl_inchi_fail", inn, e, file=sys.stderr)
            continue
        if dci and cmi and dci.strip() == cmi.strip():
            verified[inn] = m
        else:
            rejected.append((inn, dci, cmi))
    print(f"structure-verified {len(verified)} of {len(matches)}; rejected {len(rejected)}", file=sys.stderr)
    for inn, dci, cmi in rejected:
        print("REJECT", inn, "dc=", dci, "chembl=", cmi, file=sys.stderr)
    (ROOT / "tmp_fu_probe" / "drugcentral_matches_verified.json").write_text(json.dumps(verified, indent=1))


if __name__ == "__main__":
    main()
