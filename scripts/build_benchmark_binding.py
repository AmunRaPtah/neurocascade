"""Build the Phase 28 receptor binding panels (Ki + efficacy) per benchmark drug (P28.2).

Two stages, run separately:

``python scripts/build_benchmark_binding.py fetch [--cache DIR]``
    Network. Downloads the three sources into a cache directory (default
    ``<tmp>/neurocascade_p28_2``, outside the repository; never committed):

    * PDSP Ki database, full CSV, ``https://pdsp.unc.edu/databases/kiDownload/download.php``;
    * IUPHAR/BPS Guide to Pharmacology bulk ``interactions.csv`` and
      ``ligands.csv`` from ``https://www.guidetopharmacology.org/DATA/``;
    * ChEMBL REST API: every ``Ki`` activity on a human-organism target for
      each drug's parent molecule (salts included through
      ``parent_molecule_chembl_id``), the assay, document and target records
      those activities point at, and the drug's curated ``mechanism`` rows.

``python scripts/build_benchmark_binding.py build [--cache DIR]``
    Offline. Writes ``src/neurocascade/benchmark/binding/<inn>.json`` for
    every benchmark drug and ``coverage.json`` beside them.

``python scripts/build_benchmark_binding.py relink [--cache DIR]``
    Offline, needs only the cached PDSP CSV. Re-derives every committed PDSP
    measurement's reference with the current Link rules and recomputes the
    summaries it touches (P29.7).

Every inclusion rule (human assay organism, ChEMBL confidence score, PDSP name
aliases, censored values, how a per-target Ki is summarised) and the reasoning
behind it is in docs/architecture.md, section "P28.2".
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
import ssl
import statistics
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH_DIR = ROOT / "src" / "neurocascade" / "benchmark"
OUT_DIR = BENCH_DIR / "binding"
DEFAULT_CACHE = Path(tempfile.gettempdir()) / "neurocascade_p28_2"

CHEMBL = "https://www.ebi.ac.uk/chembl/api/data/"
PDSP_URL = "https://pdsp.unc.edu/databases/kiDownload/download.php"
GTOPDB_URL = "https://www.guidetopharmacology.org/DATA/"

# Retrieval dates, recorded in every output file. Set to the date the fetch
# stage was run for the committed panels.
RETRIEVED = "2026-10-02"
CHEMBL_RELEASE = "ChEMBL_37 (2026-05-01)"

# The ChEMBL molecule queried when it is not drugs.json's chembl_id, with why.
CHEMBL_QUERY_OVERRIDE = {
    "flupentixol": (
        "CHEMBL54661",
        "ATC-mapped CHEMBL6066864 'FLUPENTIXOL' carries no structure and no "
        "activities. The benchmark record's CAS (53772-82-0) identifies the "
        "cis(Z) isomer (P28.1), which is ChEMBL CHEMBL54661 'Z-FLUPENTIXOL' and "
        "GtoPdb ligand 948. The Lundbeck FLUANXOL product monograph (Canada) "
        "states commercial flupentixol dihydrochloride is an approximately 1:1 "
        "cis:trans mixture whose trans isomer is pharmacologically inactive, "
        "and the depot is cis(Z)-flupentixol decanoate. The panel is therefore "
        "the active cis(Z) isomer's; Phase 29 must take the oral isomer ratio "
        "into account when converting dose to active-isomer exposure",
    ),
}

# PDSP ligand names accepted for a drug besides its exact INN
# (case-insensitive). Each is a spelling variant, a salt, or a synonym for the
# same stereochemical entity. Names for a different stereoisomer of the drug
# (e.g. "(+)Mianserin" for racemic mianserin) are deliberately absent.
PDSP_ALIASES = {
    "chlorpromazine": ["CPZ"],
    "chlorprothixene": ["Chlorprothixene (cis)"],  # marketed substance is cis(Z)
    "clomipramine": ["CHLORIMIPRAMINE", "CLOMIPRIMINE"],
    "desvenlafaxine": ["O-Desmethylvenlafaxine"],
    "dibenzepin": ["Dibenzepine"],
    "doxepin": ["Doxepine"],
    "duloxetine": ["s-Duloxetine"],  # marketed duloxetine is the (S) enantiomer
    "escitalopram": ["1, (S)-citalopram"],
    "esketamine": ["(S)-(+)-ketamine"],
    "esmethadone": ["Methadone,(S)(+)"],
    "flupentixol": ["FLUPENTHIXOL, Alpha", "cis-flupenthixol"],  # alpha = cis(Z)
    "iproniazide": ["Iproniazid"],
    "levosulpiride": ["SULPIRIDE,(-)", "Sulpiride-S", "Sulpiride-l"],
    "lurasidone": ["Lurasidone HCl"],
    "melperone": ["Melperone HCl"],
    "mianserin": ["(+-)Mianserin", "mianserine"],
    "mirtazapine": ["mirtazepine"],
    "moclobemide": ["moclobemide, MCL"],
    "nortriptyline": ["NORTRYPTYLINE"],
    "paliperidone": ["9-OH-risperidone"],  # paliperidone is racemic 9-hydroxyrisperidone
    "periciazine": ["PROPERICIAZINE"],
    "sulpiride": ["Sulpirid"],
    "toloxatone": ["toloxatone, TOL"],
    "trifluoperazine": ["Trifluperazine"],
    "trifluperidol": ["TRIFLUORPERIDOL"],
    "triflupromazine": ["Trifluopromazine"],
    "trimipramine": ["D,L-Trimipramine"],
    "zimeldine": ["ZIMELIDINE", "Zimelidine (Z-isomer)"],  # zimeldine is the (Z) isomer
    "zuclopenthixol": ["cis-Clopenthixol"],  # zuclopenthixol is cis(Z)-clopenthixol
}

# PDSP receptor names -> human gene symbols. Only names that identify one gene
# product (or one defined subunit combination) are mapped; generic names
# ("alpha1", "5-HT2", "Cholinergic, muscarinic", "GABA A", "Sigma") are not,
# and rows carrying them are counted as unmapped rather than guessed.
PDSP_TARGETS = {
    "5-HT Transporter": "SLC6A4", "SERT": "SLC6A4", "Serotonin Transporter (SERT)": "SLC6A4",
    "5-HT1A": "HTR1A", "Serotonin 5-HT1A": "HTR1A", "h5-HT1A": "HTR1A",
    "5-HT1B": "HTR1B", "5-HT1D": "HTR1D", "5-HT1E": "HTR1E", "5-HT1F": "HTR1F",
    "5-HT2A": "HTR2A", "Serotonin 5-HT2A": "HTR2A", "h5-HT2A": "HTR2A",
    "5-HT2B": "HTR2B", "Serotonin 5-HT2B": "HTR2B", "h5-HT2B": "HTR2B",
    "5-HT2C": "HTR2C", "Serotonin 5-HT2C": "HTR2C", "5-HT2C-INI": "HTR2C", "5-HT2c VGI": "HTR2C",
    "5-HT3": "HTR3A", "5-HT4L": "HTR4", "5-HT5a": "HTR5A", "5-HT6": "HTR6",
    "5-HT7": "HTR7", "Serotonin 5-HT7": "HTR7", "5-HT7L": "HTR7", "5-HT7S": "HTR7", "5-HT7b": "HTR7",
    "ADENOSINE A1": "ADORA1", "ADENOSINE A2a": "ADORA2A", "ADENOSINE A3": "ADORA3",
    "Alpha1D": "ADRA1D", "adrenergic Alpha1D": "ADRA1D",
    "adrenergic Alpha1A": "ADRA1A", "alpha1A": "ADRA1A",
    "adrenergic Alpha1B": "ADRA1B", "alpha1B": "ADRA1B", "halpha1B-adrenergic": "ADRA1B",
    "Alpha2A": "ADRA2A", "adrenergic Alpha2A": "ADRA2A", "noradrenaline-alpha2A": "ADRA2A",
    "Alpha2B": "ADRA2B", "adrenergic Alpha2B": "ADRA2B",
    "Alpha2C": "ADRA2C", "adrenergic Alpha2C": "ADRA2C", "halpha2C-adrenergic": "ADRA2C",
    "noradrenaline-alpha2C": "ADRA2C",
    "adrenergic Beta1": "ADRB1", "beta1": "ADRB1",
    "adrenergic Beta2": "ADRB2", "beta2": "ADRB2",
    "adrenergic Beta3": "ADRB3", "beta3": "ADRB3",
    "CANNABINOID CB1": "CNR1", "CANNABINOID CB2": "CNR2",
    "Cholecystokinin A": "CCKAR", "Cholecystokinin B": "CCKBR",
    "Cholinergic, muscarinic M1": "CHRM1", "Muscarinic M1": "CHRM1", "hM1": "CHRM1",
    "Cholinergic, muscarinic M2": "CHRM2", "Cholinergic, muscarinic M3": "CHRM3",
    "Cholinergic, muscarinic M4": "CHRM4", "Cholinergic, muscarinic M5": "CHRM5",
    "Cholinergic, Nicotinic Alpha2Beta2": "CHRNA2+CHRNB2",
    "Cholinergic, Nicotinic Alpha2Beta4": "CHRNA2+CHRNB4",
    "Cholinergic, Nicotinic Alpha3Beta2": "CHRNA3+CHRNB2",
    "Cholinergic, Nicotinic Alpha3Beta4": "CHRNA3+CHRNB4",
    "Cholinergic, Nicotinic Alpha4Beta2": "CHRNA4+CHRNB2",
    "Cholinergic, Nicotinic Alpha4Beta4": "CHRNA4+CHRNB4",
    "Cholinergic, Nicotinic Alpha7": "CHRNA7",
    "D1": "DRD1", "DOPAMINE D1": "DRD1", "Dopamine D1A": "DRD1",
    "D2": "DRD2", "DOPAMINE D2": "DRD2", "D2L": "DRD2", "D2S": "DRD2", "hD2L": "DRD2",
    "DOPAMINE D2 Long": "DRD2", "DOPAMINE D2 Short": "DRD2", "Dopamine D2A": "DRD2",
    "D3": "DRD3", "DOPAMINE D3": "DRD3", "hD3": "DRD3",
    "D4": "DRD4", "DOPAMINE D4": "DRD4", "DOPAMINE D4.2": "DRD4", "DOPAMINE D4.4": "DRD4",
    "D5": "DRD5", "DOPAMINE D5": "DRD5",
    "DAT": "SLC6A3", "Dopamine Transporter": "SLC6A3",
    "Norepinephrine transporter": "SLC6A2", "NRI": "SLC6A2",
    "GABA A Alpha1Beta1Gamma2": "GABRA1+GABRB1+GABRG2",
    "GABA A Alpha1Beta3Gamma2": "GABRA1+GABRB3+GABRG2", "GABAA alpha1beta3gamma2": "GABRA1+GABRB3+GABRG2",
    "alpha1beta3gamma2": "GABRA1+GABRB3+GABRG2",
    "GABA A Alpha2Beta1Gamma2": "GABRA2+GABRB1+GABRG2",
    "GABA A Alpha2Beta3Gamma2": "GABRA2+GABRB3+GABRG2", "GABAA alpha2beta3gamma2": "GABRA2+GABRB3+GABRG2",
    "alpha2beta3gamma2": "GABRA2+GABRB3+GABRG2",
    "GABA A Alpha3Beta1Gamma2": "GABRA3+GABRB1+GABRG2",
    "GABA A Alpha3Beta3Gamma2": "GABRA3+GABRB3+GABRG2", "GABAA alpha3beta3gamma2": "GABRA3+GABRB3+GABRG2",
    "alpha3beta3gamma2": "GABRA3+GABRB3+GABRG2",
    "GABA A Alpha5Beta3Gamma2": "GABRA5+GABRB3+GABRG2", "GABAA alpha5beta3gamma2": "GABRA5+GABRB3+GABRG2",
    "alpha5beta3gamma2": "GABRA5+GABRB3+GABRG2",
    "GABA A Alpha6Beta3Gamma2": "GABRA6+GABRB3+GABRG2", "GABAA alpha6beta3gamma2": "GABRA6+GABRB3+GABRG2",
    "H1": "HRH1", "HISTAMINE H1": "HRH1", "hH1": "HRH1",
    "H2": "HRH2", "HISTAMINE H2": "HRH2", "H3": "HRH3", "HISTAMINE H3": "HRH3",
    "H4": "HRH4", "HISTAMINE H4": "HRH4",
    "HERG": "KCNH2", "IKr": "KCNH2",
    "KOR": "OPRK1", "OPIATE Kappa": "OPRK1", "OPIATE Delta": "OPRD1", "OPIATE Mu": "OPRM1",
    "NPY-Y1": "NPY1R", "Neurokinin NK1": "TACR1", "Neurokinin NK2": "TACR2", "Neurokinin NK3": "TACR3",
    "Neurotensin 1": "NTSR1", "Nociceptin receptor": "OPRL1",
    "PAF Platelet activating factor": "PTAFR", "PKC alpha": "PRKCA", "PKC gamma": "PRKCG",
    "Prostaglandin E3": "PTGER3", "Prostaglandin E4": "PTGER4",
    "Sigma 1": "SIGMAR1", "Sigma 2": "TMEM97",
    "Thromboxane A2": "TBXA2R",
    "VASOPRESSIN V1A": "AVPR1A", "VASOPRESSIN V1B": "AVPR1B", "Vasopressin V3": "AVPR1B",
    "VASOPRESSIN V2": "AVPR2",
}

# PDSP rows whose Link is not a PubMed URL, resolved to the paper the Link
# points at (P29.7). Keyed by PDSP's Reference string. The pre-P29.7 builder
# took the first 6-9 digit run of any Link as a PMID; those digits, kept here
# as `pdsp_id`, are publisher ids, not PDSP record ids and not PMIDs: two did
# not exist in PubMed and the rest resolved to unrelated papers. Each paper
# below was matched on its DOI or publisher pii and then verified by title and
# authors against PubMed esummary on 2026-10-15. `named` lists the benchmark
# drugs its PubMed title or abstract names; any other drug citing it gets a
# pmid_note built from `scope`.
PDSP_PAPERS = {
    "Schmidt, AW. et, al. 2001": {
        "pmid": "11513838", "doi": "10.1016/s0014-2999(01)01188-8", "pdsp_id": "001429990",
        "citation": "Schmidt AW, Lebel LA, Howard HR Jr, Zorn SH (2001) Eur J Pharmacol 425:197-201",
        "named": ["ziprasidone"],
        "scope": "Ziprasidone human receptor-binding profile; MeSH lists clozapine, haloperidol, "
                 "olanzapine, quetiapine and risperidone as substances, the abstract does not.",
    },
    "Zajdel, Pawel et. al, 2018": {
        "pmid": "29407591", "doi": "10.1016/j.ejmech.2018.01.002", "pdsp_id": "022352341",
        "citation": "Zajdel P, Kos T, Marciniec K, et al. (2018) Eur J Med Chem 145:790-804",
        "named": [],
        "scope": "Medicinal-chemistry series of novel azinesulfonamides of cyclic amines as "
                 "potential antipsychotics.",
    },
    "Ablordeppey S, et al., 2006": {
        "pmid": "16621538", "doi": "10.1016/j.bmcl.2006.03.057", "pdsp_id": "0960894",
        "citation": "Ablordeppey SY, Lyles-Eggleston M, Bricker B, Zhang W, Zhu X, Goodman C, "
                    "Roth BL (2006) Bioorg Med Chem Lett 16:3219-23",
        "named": ["haloperidol"],
        "scope": "Binding of (+)-SYA 09, a pyrrolidine analog of haloperidol; MeSH lists clozapine "
                 "as a substance, the abstract does not.",
    },
    "Panczyk K, et al., 2018": {
        "pmid": "29730027", "doi": "10.1016/j.bmcl.2018.04.059", "pdsp_id": "0960894",
        "citation": "Pańczyk K, Pytka K, Jakubczyk M, et al. (2018) Bioorg Med Chem Lett 28:2039-2049",
        "named": [],
        "scope": "Medicinal-chemistry series of N-(phenoxyalkyl)piperazines with 5-HT1A, 5-HT2A, "
                 "5-HT6, 5-HT7, D2 and alpha1 affinities.",
    },
    "Yarravarapu N, et. al., 2018": {
        "pmid": "30266542", "doi": "10.1016/j.bmcl.2018.09.029", "pdsp_id": "0960894",
        "citation": "Yarravarapu N, Geffert L, Surratt CK, Cascio M, Lapinsky DJ (2018) "
                    "Bioorg Med Chem Lett 28:3431-3435",
        "named": [],
        "scope": "Photoaffinity ligands derived from escitalopram, which the abstract names "
                 "as (S)-citalopram and measures at hSERT (Ki 1.77 nM).",
    },
    "Van der Mey. et, al. 2006": {
        "pmid": "16517171", "doi": "10.1016/j.bmc.2006.02.029", "pdsp_id": "096808960",
        "citation": "Van der Mey M, Windhorst AD, Klok RP, et al. (2006) Bioorg Med Chem 14:4526-34",
        "named": ["mirtazapine"],
        "scope": "Alpha2-adrenoceptor PET ligand [11C]R107474; reports mirtazapine's "
                 "receptor binding profile.",
    },
    "Keck, Paul e. et. al, 2003": {
        "pmid": "12665420", "doi": "10.1517/13543784.12.4.655", "pdsp_id": "13543784",
        "citation": "Keck PE Jr, McElroy SL (2003) Expert Opin Investig Drugs 12:655-62",
        "named": ["aripiprazole"],
        "scope": "Review of aripiprazole pharmacology (secondary source).",
    },
    "Gassaway, MM.  et. al. 2014": {
        "pmid": "25026323", "doi": "10.1038/tp.2014.30", "pdsp_id": "4119213",
        "citation": "Gassaway MM, Rives ML, Kruegel AC, Javitch JA, Sames D (2014) "
                    "Transl Psychiatry 4:e411",
        "named": ["tianeptine"],
        "scope": "Tianeptine as a mu-opioid receptor agonist.",
    },
    "Clayton, Terry et. al. 2015": {
        "pmid": "26682068", "doi": "10.1155/2015/430248", "pdsp_id": "430248",
        "citation": "Clayton T, Poe MM, Rallapalli S, et al. (2015) Int J Med Chem 2015:430248",
        "named": [],
        "scope": "Review of the alpha5 GABA-A benzodiazepine-site pharmacophore. Its full text "
                 "(PMC4657098) tabulates alprazolam and diazepam Ki at recombinant "
                 "alpha(x)beta3gamma2 receptors, matching the PDSP rows.",
    },
    "Song, Gang et., al.  2017": {
        "pmid": "28239660", "doi": "10.1172/jci.insight.91456", "pdsp_id": "5313063",
        "citation": "Song G, Poon CS (2017) JCI Insight 2:e91456",
        "named": [],
        "scope": "Alpha2-adrenergic blockade (yohimbine) in obstructive sleep apnea.",
    },
    "Davies M, et al. 2005": {
        "pmid": "16314884", "doi": "10.1038/sj.tpj.6500342", "pdsp_id": "6500342",
        "citation": "Davies MA, Setola V, Strachan RT, Sheffler DJ, Salay E, Hufeisen SJ, "
                    "Roth BL (2006) Pharmacogenomics J 6:42-51",
        "named": ["aripiprazole", "clozapine", "olanzapine", "quetiapine", "risperidone",
                  "ziprasidone"],
        "scope": "Atypical antipsychotics at non-synonymous h5-HT2A SNP variants.",
    },
    "Silvestre, JS. et, al. 2005": {
        "pmid": "16082416", "doi": "10.1358/mf.2005.27.5.908643", "pdsp_id": "908643",
        "citation": "Silvestre JS, Prous J (2005) Methods Find Exp Clin Pharmacol 27:289-304",
        "named": [],
        "scope": "Compilation, not a primary measurement: pKi of 25 typical and atypical "
                 "antipsychotics at 21 human receptors, taken from Prous Science Integrity, "
                 "regressed against type 2 diabetes risk.",
    },
    "Seeman, Philip  2001": {
        "pmid": None, "doi": "10.1016/S1566-2772(00)00007-4", "pdsp_id": "93617739",
        "citation": "Seeman P (2001) Antipsychotic drugs, dopamine receptors, and "
                    "schizophrenia. Clin Neurosci Res 1:53-60",
        "named": [],
        "scope": "Not indexed in PubMed: esearch on the DOI and on the title returned nothing "
                 "(2026-10-15); paper identified from the ScienceDirect link's cover date and "
                 "Crossref. pdsp_id is the link's ScienceDirect _alid, not a PMID.",
    },
}

# Action vocabulary: source label -> normalised action type.
ACTION_NORMALISE = {
    "antagonist": "antagonist", "inverse agonist": "inverse agonist",
    "agonist": "agonist", "full agonist": "agonist", "partial agonist": "partial agonist",
    "inhibitor": "inhibitor", "inhibition": "inhibitor", "irreversible inhibition": "inhibitor",
    "irreversible inhibitor": "inhibitor", "blocker": "blocker", "pore blocker": "blocker",
    "channel blocker": "blocker", "negative allosteric modulator": "negative allosteric modulator",
    "negative": "negative allosteric modulator",
    "positive allosteric modulator": "positive allosteric modulator",
    "positive": "positive allosteric modulator", "potentiation": "positive allosteric modulator",
    "allosteric antagonist": "negative allosteric modulator",
    "modulator": "modulator", "releasing agent": "releasing agent", "opener": "opener",
    "biased agonist": "agonist", "activation": "agonist", "neutral": "neutral antagonist",
    "competitive": "antagonist", "non-competitive": "antagonist", "activator": "activator",
    "gating inhibitor": "gating inhibitor", "slows inactivation": "gating inhibitor",
    "voltage-dependent inhibition": "blocker", "feedback inhibition": "inhibitor",
    "subunit-specific": None, "antibody": None, "fusion protein": None, "binding": None, "none": None, "unknown": None, "mixed": None,
    "": None,
}


# ---------------------------------------------------------------- fetching


def _ctx():
    # pdsp.unc.edu does not present a chain this image's CA bundle verifies
    # (the same failure P8.1 recorded on 2026-08-14). The payload is public
    # reference data and is sha256-recorded in every output file.
    return ssl._create_unverified_context()


def _download(url: str, timeout: int = 600) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "neurocascade-P28.2"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=timeout, context=_ctx()) as r:
                return r.read()
        except Exception:  # noqa: BLE001 - retry any transport error
            if attempt == 4:
                raise
            time.sleep(5 * (attempt + 1))
    raise AssertionError("unreachable")


def _chembl_all(cache: Path, path: str, params: dict, key: str) -> list[dict]:
    """Every page of a ChEMBL list endpoint, cached by request."""
    q = urllib.parse.urlencode(sorted(params.items()))
    fn = cache / "chembl" / (hashlib.sha1(f"{path}?{q}".encode()).hexdigest() + ".json")
    if fn.exists():
        return json.loads(fn.read_text())
    out: list[dict] = []
    url = f"{CHEMBL}{path}.json?{q}&limit=1000"
    while url:
        page = json.loads(_download(url, timeout=300))
        out.extend(page[key])
        nxt = page["page_meta"]["next"]
        url = f"https://www.ebi.ac.uk{nxt}" if nxt else None
    fn.parent.mkdir(parents=True, exist_ok=True)
    fn.write_text(json.dumps(out))
    return out


def _chembl_by_ids(cache: Path, path: str, idfield: str, ids, key: str) -> dict:
    ids = sorted(set(ids))
    out = {}
    for i in range(0, len(ids), 50):
        chunk = ids[i : i + 50]
        for rec in _chembl_all(cache, path, {f"{idfield}__in": ",".join(chunk)}, key):
            out[rec[idfield]] = rec
    return out


def chembl_query_id(drug: dict) -> str:
    return CHEMBL_QUERY_OVERRIDE.get(drug["inn"], (drug["chembl_id"], None))[0]


def fetch(cache: Path) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    for name, url in (
        ("pdsp.csv", PDSP_URL),
        ("gtopdb_interactions.csv", GTOPDB_URL + "interactions.csv"),
        ("gtopdb_ligands.csv", GTOPDB_URL + "ligands.csv"),
    ):
        if not (cache / name).exists():
            (cache / name).write_bytes(_download(url))
            print("downloaded", name, file=sys.stderr)
    drugs = json.loads((BENCH_DIR / "drugs.json").read_text())["drugs"]
    acts_all: dict[str, list] = {}
    for d in drugs:
        cid = chembl_query_id(d)
        acts_all[d["inn"]] = _chembl_all(
            cache, "activity",
            {"parent_molecule_chembl_id": cid, "standard_type": "Ki",
             "target_organism": "Homo sapiens"},
            "activities",
        )
        _chembl_all(cache, "mechanism", {"parent_molecule_chembl_id": cid}, "mechanisms")
        print(d["inn"], cid, len(acts_all[d["inn"]]), file=sys.stderr)
    acts = [a for v in acts_all.values() for a in v]
    _chembl_by_ids(cache, "assay", "assay_chembl_id", [a["assay_chembl_id"] for a in acts], "assays")
    _chembl_by_ids(cache, "document", "document_chembl_id",
                   [a["document_chembl_id"] for a in acts], "documents")
    tids = {a["target_chembl_id"] for a in acts}
    for d in drugs:
        for m in _chembl_all(cache, "mechanism", {"parent_molecule_chembl_id": chembl_query_id(d)},
                             "mechanisms"):
            if m.get("target_chembl_id"):
                tids.add(m["target_chembl_id"])
    _chembl_by_ids(cache, "target", "target_chembl_id", tids, "targets")


# ---------------------------------------------------------------- building


def _read_gtopdb(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    version = lines[0].strip('"').lstrip("# ")
    rows = list(csv.DictReader(io.StringIO("\n".join(lines[1:]))))
    for r in rows:
        r["_version"] = version
    return rows


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _target_key(t: dict) -> tuple[str, list[str], list[str]] | None:
    """Gene-symbol key, gene symbols and UniProt accessions for a ChEMBL target."""
    genes, accs = [], []
    for c in t.get("target_components") or []:
        accs.append(c.get("accession"))
        syms = [s["component_synonym"] for s in c.get("target_component_synonyms") or []
                if s["syn_type"] == "GENE_SYMBOL"]
        if not syms:
            return None
        genes.append(syms[0].upper())
    if not genes:
        return None
    genes_sorted = sorted(set(genes))
    return "+".join(genes_sorted), genes_sorted, sorted(a for a in set(accs) if a)


def _norm_action(label: str | None) -> str | None:
    if label is None:
        return None
    return ACTION_NORMALISE.get(label.strip().lower(), label.strip().lower())


def _gtopdb_action(type_: str, action: str) -> str | None:
    """GtoPdb prints a ligand 'Type' and an 'Action'. The Action is the more
    specific of the two except where it is empty or a bare 'Binding'/'None';
    'Positive'/'Negative' only mean allosteric modulation under the
    'Allosteric modulator' type."""
    t, a = type_.strip().lower(), action.strip().lower()
    if t == "allosteric modulator" and a in ("positive", "potentiation", "activation"):
        return "positive allosteric modulator"
    if t == "allosteric modulator" and a in ("negative", "inhibition"):
        return "negative allosteric modulator"
    if a in ("", "none", "binding", "unknown", "mixed"):
        a = t
    if a == "positive":
        return "activator"
    return _norm_action(a)


def _pmid_from_link(link: str) -> str | None:
    """The PMID in a PDSP Link, which is only a PMID when the link is PubMed's.

    Any other URL's digits are a publisher's id (an Elsevier pii's ISSN
    prefix, a PMC id, a Prous RefId, a Hindawi article number) and are not
    taken (P29.7).
    """
    link = link or ""
    if "ncbi.nlm.nih.gov" not in link or "/pmc/" in link:
        return None
    m = re.search(r"(\d{6,9})", link)
    return m.group(1) if m else None


def _pdsp_ref(reference: str, link: str, inn: str) -> tuple[str, dict]:
    """Reference key and record for one PDSP row (P29.7)."""
    reference = reference.strip()
    pmid = _pmid_from_link(link)
    if pmid:
        return f"PMID:{pmid}", {"pmid": pmid, "doi": None, "citation": reference}
    paper = PDSP_PAPERS.get(reference)
    if paper is None:
        return f"PDSP_REF:{reference}", {"pmid": None, "doi": None, "citation": reference}
    rec = {"pmid": paper["pmid"], "doi": paper["doi"], "citation": paper["citation"],
           "pdsp_reference": reference, "pdsp_id": paper["pdsp_id"], "pdsp_link": link.strip()}
    if paper["pmid"] and inn not in paper["named"]:
        rec["pmid_note"] = (f"{paper['scope']} The PDSP Ki database attributes this drug's "
                            f"Ki rows to this paper; {inn} is not named in its PubMed "
                            f"title or abstract.")
    if not paper["pmid"]:
        rec["pmid_note"] = paper["scope"]
    key = f"PMID:{paper['pmid']}" if paper["pmid"] else f"PDSP_REF:{reference}"
    return key, rec


def _note_unnamed_papers(refs: dict[str, dict], inn: str) -> None:
    """Note a PDSP_PAPERS paper cited for a drug its abstract does not name.

    `_pdsp_ref` notes the PDSP rows; this covers the same paper reaching a
    drug through another source (ChEMBL cites Panczyk 2018 for hydroxyzine),
    so every citation of these twelve PMIDs is either named or noted (P29.7).
    """
    by_pmid = {p["pmid"]: p for p in PDSP_PAPERS.values() if p["pmid"]}
    for rec in refs.values():
        paper = by_pmid.get(rec.get("pmid"))
        if paper and inn not in paper["named"] and not rec.get("pmid_note"):
            rec["pmid_note"] = (f"{paper['scope']} The source database attributes this "
                                f"drug's rows to this paper; {inn} is not named in its "
                                f"PubMed title or abstract.")


def build(cache: Path) -> None:
    drugs = json.loads((BENCH_DIR / "drugs.json").read_text())["drugs"]
    pdsp_path = cache / "pdsp.csv"
    pdsp = list(csv.DictReader(io.StringIO(pdsp_path.read_bytes().decode("latin-1"))))
    gi_path, gl_path = cache / "gtopdb_interactions.csv", cache / "gtopdb_ligands.csv"
    g_inter, g_lig = _read_gtopdb(gi_path), _read_gtopdb(gl_path)
    gtopdb_version = g_inter[0]["_version"]
    lig_by_ik = {r["InChIKey"]: r for r in g_lig if r["InChIKey"]}
    lig_by_chembl = {r["ChEMBL ID"]: r for r in g_lig if r["ChEMBL ID"]}
    inter_by_lig = defaultdict(list)
    for r in g_inter:
        inter_by_lig[r["Ligand ID"]].append(r)
    pdsp_by_name = defaultdict(list)
    for r in pdsp:
        pdsp_by_name[r[" Ligand Name"].strip().lower()].append(r)

    acts_cache = {}
    for d in drugs:
        acts_cache[d["inn"]] = _chembl_all(
            cache, "activity",
            {"parent_molecule_chembl_id": chembl_query_id(d), "standard_type": "Ki",
             "target_organism": "Homo sapiens"}, "activities")
    all_acts = [a for v in acts_cache.values() for a in v]
    assays = _chembl_by_ids(cache, "assay", "assay_chembl_id",
                            [a["assay_chembl_id"] for a in all_acts], "assays")
    docs = _chembl_by_ids(cache, "document", "document_chembl_id",
                          [a["document_chembl_id"] for a in all_acts], "documents")
    mechs = {d["inn"]: _chembl_all(cache, "mechanism",
                                   {"parent_molecule_chembl_id": chembl_query_id(d)}, "mechanisms")
             for d in drugs}
    tids = {a["target_chembl_id"] for a in all_acts}
    tids |= {m["target_chembl_id"] for v in mechs.values() for m in v if m.get("target_chembl_id")}
    targets = _chembl_by_ids(cache, "target", "target_chembl_id", tids, "targets")

    provenance = {
        "retrieved": RETRIEVED,
        "pdsp": {"url": PDSP_URL, "sha256": _sha(pdsp_path), "rows": len(pdsp)},
        "gtopdb": {"url": GTOPDB_URL + "interactions.csv", "version": gtopdb_version,
                   "sha256": _sha(gi_path)},
        "chembl": {"api": CHEMBL, "release": CHEMBL_RELEASE},
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    coverage_rows = []
    for d in drugs:
        panel = _build_one(d, acts_cache[d["inn"]], assays, docs, targets, mechs[d["inn"]],
                           pdsp_by_name, lig_by_ik, lig_by_chembl, inter_by_lig)
        panel["provenance"] = provenance
        fn = OUT_DIR / f"{d['inn'].replace(' ', '_')}.json"
        fn.write_text(json.dumps(panel, indent=1, sort_keys=True) + "\n")
        coverage_rows.append({
            "inn": d["inn"], "stratum": d["stratum"], "status": panel["status"],
            "n_targets_with_ki": len(panel["targets"]),
            "n_targets_with_action": sum(1 for t in panel["targets"].values() if t["action_types"]),
            "n_targets_with_family_action_only": sum(
                1 for t in panel["targets"].values()
                if not t["action_types"] and t["action_types_via_family"]),
            "n_ki_measurements": sum(t["n_measurements"] for t in panel["targets"].values()),
            "n_mechanism_targets": len(panel["mechanism_targets"]),
            # a family-level mechanism target counts as covered when the
            # panel has a Ki at one of its members
            "n_mechanism_targets_with_ki": sum(
                1 for k in panel["mechanism_targets"]
                if any(set(t.split("+")) <= set(k.split("+")) for t in panel["targets"])),
        })
    _write_coverage(coverage_rows, provenance)


def _build_one(d, acts, assays, docs, targets, mechs, pdsp_by_name, lig_by_ik,
               lig_by_chembl, inter_by_lig) -> dict:
    inn = d["inn"]
    refs: dict[str, dict] = {}
    meas: dict[str, list] = defaultdict(list)  # target key -> measurements
    tinfo: dict[str, dict] = {}
    excluded = defaultdict(int)
    inactive: dict[str, list] = defaultdict(list)

    def add_ref(key, rec):
        # The first source to cite a paper sets its record; a later one only
        # adds fields it lacks (a PDSP row's pdsp_id and pmid_note, P29.7).
        for k, v in rec.items():
            refs.setdefault(key, {}).setdefault(k, v)

    # --- ChEMBL
    for a in acts:
        assay = assays.get(a["assay_chembl_id"]) or {}
        t = targets.get(a["target_chembl_id"]) or {}
        if a.get("data_validity_comment"):
            excluded["chembl_data_validity_flag"] += 1
            continue
        if a.get("potential_duplicate"):
            excluded["chembl_potential_duplicate"] += 1
            continue
        if str(assay.get("assay_tax_id")) != "9606":
            excluded["chembl_assay_organism_not_human"] += 1
            continue
        ttype = t.get("target_type")
        conf = assay.get("confidence_score")
        if not ((ttype == "SINGLE PROTEIN" and conf == 9)
                or (ttype in ("PROTEIN COMPLEX", "PROTEIN COMPLEX GROUP") and conf == 7)):
            excluded["chembl_target_not_direct_single_protein_or_complex"] += 1
            continue
        tk = _target_key(t)
        if tk is None:
            excluded["chembl_target_without_gene_symbol"] += 1
            continue
        if a.get("standard_value") in (None, ""):
            # DrugMatrix screens print no Ki when inhibition at 10 uM was
            # under 50%; the printed statement is kept, no number is made up.
            if a.get("activity_comment") or a.get("standard_text_value"):
                inactive[tk[0]].append({
                    "statement": a.get("activity_comment") or a.get("standard_text_value"),
                    "source": "ChEMBL", "chembl_document": a["document_chembl_id"],
                    "record": f"activity {a['activity_id']} / assay {a['assay_chembl_id']}"})
            excluded["chembl_no_ki_value"] += 1
            continue
        if a.get("standard_units") != "nM":
            excluded["chembl_not_nM"] += 1
            continue
        rel = a.get("standard_relation")
        if rel not in ("=", ">", ">="):
            excluded["chembl_relation_" + str(rel)] += 1
            continue
        value = float(a["standard_value"])
        if value <= 0:
            excluded["chembl_nonpositive"] += 1
            continue
        doc = docs.get(a["document_chembl_id"]) or {}
        pmid = doc.get("pubmed_id")
        rkey = f"PMID:{pmid}" if pmid else f"CHEMBL_DOC:{a['document_chembl_id']}"
        add_ref(rkey, {"pmid": str(pmid) if pmid else None, "doi": doc.get("doi"),
                       "citation": f"{doc.get('authors') or ''} ({doc.get('year')}) "
                                   f"{doc.get('journal') or doc.get('doc_type') or ''} "
                                   f"{doc.get('volume') or ''}:{doc.get('first_page') or ''}".strip(),
                       "chembl_document": a["document_chembl_id"]})
        key, genes, accs = tk
        tinfo.setdefault(key, {"gene_symbols": genes, "uniprot": accs,
                               "name": t.get("pref_name")})
        meas[key].append({"ki_nM": value, "relation": "=" if rel == "=" else ">",
                          "source": "ChEMBL", "ref": rkey,
                          "record": f"activity {a['activity_id']} / assay {a['assay_chembl_id']}"})

    # --- PDSP
    names = {inn.lower()} | {n.lower() for n in PDSP_ALIASES.get(inn, [])}
    pdsp_names_used = set()
    for n in sorted(names):
        for r in pdsp_by_name.get(n, []):
            species = r["species"].strip()
            if species != "HUMAN":
                excluded["pdsp_species_not_human"] += 1
                continue
            gene = PDSP_TARGETS.get(r["Name"].strip())
            if gene is None:
                excluded["pdsp_receptor_name_unmapped"] += 1
                continue
            note = r["ki Note"].strip()
            if note not in ("", ">"):
                excluded["pdsp_relation_" + note] += 1
                continue
            try:
                value = float(r["ki Val"])
            except ValueError:
                excluded["pdsp_value_unparseable"] += 1
                continue
            if value <= 0:
                excluded["pdsp_nonpositive"] += 1
                continue
            rkey, rec = _pdsp_ref(r["Reference"], r["Link"], inn)
            add_ref(rkey, rec)
            genes = gene.split("+")
            tinfo.setdefault(gene, {"gene_symbols": genes, "uniprot": [], "name": r["Name"].strip()})
            pdsp_names_used.add(r[" Ligand Name"].strip())
            meas[gene].append({"ki_nM": value, "relation": "=" if note == "" else ">",
                               "source": "PDSP", "ref": rkey,
                               "record": f"PDSP Ki ID {r['Number']} ({r['source'].strip()}, "
                                         f"hot ligand {r['Hotligand'].strip()})"})

    # --- GtoPdb
    lig = lig_by_ik.get(d.get("standard_inchikey") or "") or lig_by_chembl.get(chembl_query_id(d))
    actions: dict[str, list] = defaultdict(list)
    if lig is not None:
        for r in inter_by_lig[lig["Ligand ID"]]:
            if r["Target Species"] != "Human" or not r["Target Gene Symbol"]:
                continue
            genes = sorted({g.strip().upper() for g in r["Target Gene Symbol"].split("|") if g.strip()})
            key = "+".join(genes)
            pmids = [p for p in r["PubMed ID"].split("|") if p.strip()]
            rkey = (f"PMID:{pmids[0]}" if len(pmids) == 1
                    else f"GTOPDB:ligand {lig['Ligand ID']} target {r['Target ID']}")
            add_ref(rkey, {"pmid": pmids[0] if len(pmids) == 1 else None, "doi": None,
                           "citation": (f"GtoPdb interaction, ligand {lig['Ligand ID']} "
                                        f"({lig['Name']}), target {r['Target ID']}; primary "
                                        f"PMIDs {','.join(pmids) or 'none listed'}")})
            act = _gtopdb_action(r["Type"], r["Action"])
            if act:
                actions[key].append({"action": act, "source": "GtoPdb",
                                     "label": f"{r['Type']} / {r['Action']}", "ref": rkey})
            if r["Affinity Units"] != "pKi":
                continue
            tinfo.setdefault(key, {"gene_symbols": genes,
                                   "uniprot": sorted(u for u in r["Target UniProt ID"].split("|") if u),
                                   "name": re.sub(r"<[^>]+>", "", r["Target"])})
            rel = r["Original Affinity Relation"].strip()
            if rel in ("<", "<="):
                excluded["gtopdb_relation_lt"] += 1
                continue
            vals = ([r["Affinity Median"]] if r["Affinity Median"]
                    else [v for v in (r["Affinity High"], r["Affinity Low"]) if v])
            for v in vals:
                meas[key].append({"ki_nM": 10 ** (9 - float(v)),
                                  "relation": ">" if rel in (">", ">=") else "=",
                                  "source": "GtoPdb", "ref": rkey,
                                  "record": f"GtoPdb ligand {lig['Ligand ID']} target "
                                            f"{r['Target ID']} pKi {v}"})

    # --- ChEMBL mechanism (curated action type)
    for m in mechs:
        t = targets.get(m.get("target_chembl_id")) or {}
        if t.get("organism") != "Homo sapiens":
            continue
        tk = _target_key(t)
        act = _norm_action(m.get("action_type"))
        if tk is None or not act:
            continue
        pm = [r["ref_id"] for r in m.get("mechanism_refs") or [] if r["ref_type"] == "PubMed"]
        rkey = f"PMID:{pm[0]}" if pm else f"CHEMBL_MECHANISM:{m['mec_id']}"
        add_ref(rkey, {"pmid": pm[0] if pm else None, "doi": None,
                       "citation": (f"ChEMBL drug mechanism {m['mec_id']}: "
                                    f"{m.get('mechanism_of_action')}; refs "
                                    + "; ".join(f"{r['ref_type']} {r['ref_id']}"
                                                for r in m.get("mechanism_refs") or []))})
        actions[tk[0]].append({"action": act, "source": "ChEMBL mechanism",
                               "label": m.get("action_type"), "ref": rkey})

    # --- summarise per target
    out_targets = {}
    for key, ms in meas.items():
        if not any(m["relation"] == "=" for m in ms):
            continue
        out_targets[key] = {
            **tinfo[key],
            **_summarise(ms),
            "action_types": sorted({a["action"] for a in actions.get(key, [])}),
            "actions": actions.get(key, []),
            # A record made on a receptor family or on a complex this target
            # belongs to (ChEMBL "D2-like dopamine receptor antagonist", a
            # GtoPdb GABA-A family entry) says nothing subtype-specific, so it
            # is kept apart from the target's own action types.
            "action_types_via_family": sorted({
                a["action"] for fam, v in actions.items()
                if fam != key and set(key.split("+")) < set(fam.split("+")) for a in v}),
            "families": sorted(fam for fam in actions
                               if fam != key and set(key.split("+")) < set(fam.split("+"))),
            "measurements": _sorted_measurements(ms),
        }
    # actions on targets that have no exact Ki stay visible, outside the panel
    actions_without_ki = {k: v for k, v in actions.items() if k not in out_targets}
    _note_unnamed_papers(refs, inn)
    used_refs = {m["ref"] for t in out_targets.values() for m in t["measurements"]}
    used_refs |= {a["ref"] for t in out_targets.values() for a in t["actions"]}
    used_refs |= {a["ref"] for v in actions_without_ki.values() for a in v}
    status = "panel" if out_targets else "searched_and_empty"
    override = CHEMBL_QUERY_OVERRIDE.get(inn)
    return {
        "inn": inn,
        "stratum": d["stratum"],
        "chembl_id": d["chembl_id"],
        "chembl_id_queried": chembl_query_id(d),
        "entity_note": override[1] if override else None,
        "species": "Homo sapiens (assay organism and target; non-human rows excluded)",
        "status": status,
        # ChEMBL-curated mechanism-of-action targets: the targets the drug is
        # used for, which the panel either covers or visibly does not.
        "mechanism_targets": sorted({k for k, v in actions.items()
                                     if any(a["source"] == "ChEMBL mechanism" for a in v)}),
        "searched": {
            "chembl_ki_rows_returned": len(acts),
            "pdsp_ligand_names_tried": sorted(names),
            "pdsp_ligand_names_matched": sorted(pdsp_names_used),
            "gtopdb_ligand_id": lig["Ligand ID"] if lig else None,
            "excluded_counts": dict(sorted(excluded.items())),
        },
        "targets": dict(sorted(out_targets.items())),
        "actions_without_ki": dict(sorted(actions_without_ki.items())),
        "screened_inactive_without_ki": {k: sorted(v, key=lambda x: x["record"])
                                         for k, v in sorted(inactive.items())
                                         if k not in out_targets},
        "references": {k: refs[k] for k in sorted(used_refs)},
    }


def _summarise(ms: list[dict]) -> dict:
    """The per-target summary fields: the median over references of each
    reference's mean exact pKi, plus the spread (architecture.md, P28.2)."""
    exact = [m for m in ms if m["relation"] == "="]
    by_doc = defaultdict(list)
    for m in exact:
        by_doc[m["ref"]].append(-math.log10(m["ki_nM"] * 1e-9))
    pki = statistics.median(statistics.fmean(v) for v in by_doc.values())
    vals = [m["ki_nM"] for m in exact]
    return {
        "ki_nM": round(10 ** (9 - pki), 6),
        "pki_median": round(pki, 4),
        "ki_min_nM": min(vals),
        "ki_max_nM": max(vals),
        "n_measurements": len(exact),
        "n_references": len(by_doc),
        "n_censored": len(ms) - len(exact),
    }


def _sorted_measurements(ms: list[dict]) -> list[dict]:
    return sorted(ms, key=lambda m: (m["source"], m["ref"], m["ki_nM"]))


def relink_pdsp(panel: dict, pdsp_by_number: dict[str, dict]) -> dict:
    """Re-derive every PDSP measurement's reference from its PDSP row (P29.7).

    Offline repair of committed panels with the current `_pdsp_ref` rules,
    needing only the PDSP CSV. Measurement values are not touched; a target's
    summary is recomputed only when its measurements' references changed,
    since re-keying can merge one paper's PDSP rows with its ChEMBL rows.
    """
    inn, refs = panel["inn"], panel["references"]
    for t in panel["targets"].values():
        changed = False
        for m in t["measurements"]:
            if m["source"] != "PDSP":
                continue
            number = re.match(r"PDSP Ki ID (\d+) ", m["record"]).group(1)
            row = pdsp_by_number[number]
            key, rec = _pdsp_ref(row["Reference"], row["Link"], inn)
            for k, v in rec.items():
                refs.setdefault(key, {}).setdefault(k, v)
            if key != m["ref"]:
                m["ref"], changed = key, True
        if changed:
            t.update(_summarise(t["measurements"]))
            t["measurements"] = _sorted_measurements(t["measurements"])
    used = {m["ref"] for t in panel["targets"].values() for m in t["measurements"]}
    used |= {a["ref"] for t in panel["targets"].values() for a in t["actions"]}
    used |= {a["ref"] for v in panel["actions_without_ki"].values() for a in v}
    _note_unnamed_papers(panel["references"], inn)
    panel["references"] = {k: refs[k] for k in sorted(used)}
    return panel


def relink(cache: Path) -> None:
    pdsp = csv.DictReader(io.StringIO((cache / "pdsp.csv").read_bytes().decode("latin-1")))
    by_number = {r["Number"]: r for r in pdsp}
    for path in sorted(OUT_DIR.glob("*.json")):
        if path.name == "coverage.json":
            continue
        panel = json.loads(path.read_text())
        path.write_text(json.dumps(relink_pdsp(panel, by_number), indent=1, sort_keys=True) + "\n")


def _write_coverage(rows: list[dict], provenance: dict) -> None:
    covered = [r for r in rows if r["status"] == "panel"]
    n_t = [r["n_targets_with_ki"] for r in covered]
    summary = {
        "n_drugs": len(rows),
        "n_with_panel": len(covered),
        "n_searched_and_empty": len(rows) - len(covered),
        "mean_targets_per_covered_drug": round(statistics.fmean(n_t), 2) if n_t else 0.0,
        "median_targets_per_covered_drug": statistics.median(n_t) if n_t else 0,
        "n_with_any_action_type": sum(1 for r in covered if r["n_targets_with_action"]),
        "n_with_mechanism_target": sum(1 for r in rows if r["n_mechanism_targets"]),
        "n_with_ki_at_a_mechanism_target": sum(1 for r in rows if r["n_mechanism_targets_with_ki"]),
        "mean_action_typed_targets_per_covered_drug":
            round(statistics.fmean([r["n_targets_with_action"] for r in covered]), 2) if covered else 0.0,
        "searched_and_empty": sorted(r["inn"] for r in rows if r["status"] != "panel"),
        "by_stratum": {},
    }
    for s in sorted({r["stratum"] for r in rows}):
        rs = [r for r in rows if r["stratum"] == s]
        cs = [r for r in rs if r["status"] == "panel"]
        summary["by_stratum"][s] = {
            "n": len(rs), "with_panel": len(cs),
            "mean_targets_per_covered_drug":
                round(statistics.fmean([r["n_targets_with_ki"] for r in cs]), 2) if cs else 0.0,
        }
    (OUT_DIR / "coverage.json").write_text(json.dumps(
        {"summary": summary, "drugs": rows, "provenance": provenance},
        indent=1, sort_keys=True) + "\n")


def main(argv: list[str]) -> None:
    stages = {"fetch": fetch, "build": build, "relink": relink}
    if not argv or argv[0] not in stages:
        raise SystemExit(__doc__)
    cache = DEFAULT_CACHE
    if "--cache" in argv:
        cache = Path(argv[argv.index("--cache") + 1])
    stages[argv[0]](cache)


if __name__ == "__main__":
    main(sys.argv[1:])
