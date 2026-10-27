"""P29.6: the sourced microsomal-binding (fu_mic) correction on the P29.2
well-stirred CL arm.

The correction recovers the unbound intrinsic clearance from the total-protein
CLint using the Hallifax-Houston 2006 lipophilicity relationship, with a
per-drug logP (ChEMBL_37 ALOGP) and ionisation class (measured pKa, sourced per
drug in drug_physchem.json).  All of these tests are train-only and read local
JSON, so none needs held-out truth or the network.
"""

import math

import pytest

from neurocascade.benchmark import load_split
import neurocascade.benchmark.human_pk_translation as T
from neurocascade.benchmark.human_pk_translation import (
    CONSTANTS,
    cl_train_mean_baseline,
    load_physchem,
    score_train,
    score_train_fu_mic,
)

# The 15 train drugs the IVIVE CL arm scores, and the 6 held-out ones.
TRAIN_IVIVE = {
    "haloperidol", "desipramine", "chlorpromazine", "maprotiline", "mirtazapine",
    "clozapine", "risperidone", "promazine", "chlordiazepoxide", "lorazepam",
    "oxazepam", "trimipramine", "clomipramine", "trazodone", "citalopram",
}
HELDOUT_IVIVE = {
    "alprazolam", "diazepam", "fluphenazine", "imipramine", "nomifensine",
    "venlafaxine",
}


# --- sourcing (ACCEPTANCE 1) -----------------------------------------------
def test_physchem_covers_every_ivive_drug():
    drugs = load_physchem()["drugs"]
    assert TRAIN_IVIVE <= set(drugs)
    assert HELDOUT_IVIVE <= set(drugs)


def test_every_record_records_logp_and_pka_with_a_per_drug_source():
    for inn, e in load_physchem()["drugs"].items():
        assert isinstance(e["logp"], (int, float)), inn
        assert e["logp_source"] and "ChEMBL" in e["logp_source"], inn
        # pKa value AND a per-drug source string -- the ACCEPTANCE 1 gap.
        assert isinstance(e["pka"], (int, float)), inn
        assert isinstance(e.get("pka_source"), str) and e["pka_source"].strip(), inn
        assert e["ionisation_class"] in {"base", "neutral", "acid"}, inn


def test_every_pmid_source_carries_a_note_for_the_host_check():
    # The only PMID in the file (Settimo 2014, 24249037) is a multi-drug pKa
    # compilation, so the drug name is not in its title/abstract; each such
    # record must carry a pmid_note, which is what the host's PMID check wants.
    for inn, e in load_physchem()["drugs"].items():
        if "PMID" in e.get("pka_source", ""):
            assert e.get("pmid_note"), inn


def test_no_drug_in_the_arm_is_an_acid_so_all_use_logp():
    # Austin/Hallifax-Houston use logP for bases/neutrals and logD7.4 for
    # acids; every CNS drug here is a base or neutral, so logP is always used.
    for inn, e in load_physchem()["drugs"].items():
        assert e["ionisation_class"] != "acid", inn


# --- the equation ----------------------------------------------------------
def test_fu_mic_matches_hallifax_houston_closed_form():
    C = CONSTANTS.fu_mic_protein_mg_ml
    for inn, e in load_physchem()["drugs"].items():
        lp = e["logp"]
        expect = 1.0 / (1.0 + C * 10 ** (0.072 * lp * lp + 0.067 * lp - 1.126))
        got, refs, _ = T._drug_fu_mic(inn, C)
        assert got == pytest.approx(expect), inn
        assert refs  # provenance carried


def test_fu_mic_values_are_pinned_for_the_lipophilic_bases():
    C = CONSTANTS.fu_mic_protein_mg_ml
    assert C == 1.0
    assert T._drug_fu_mic("haloperidol", C)[0] == pytest.approx(0.2068, abs=1e-4)
    assert T._drug_fu_mic("chlorpromazine", C)[0] == pytest.approx(0.1066, abs=1e-4)
    assert T._drug_fu_mic("desipramine", C)[0] == pytest.approx(0.4956, abs=1e-4)


def test_fu_mic_falls_back_to_one_without_a_sourced_record():
    # A drug with no physchem row is never guessed: fu_mic = 1 (no correction).
    fu_mic, refs, _ = T._drug_fu_mic("not_a_real_drug", 1.0)
    assert fu_mic == 1.0
    assert refs == ()


# --- the result (ACCEPTANCE 2, 3, 4) ---------------------------------------
def test_frozen_default_arm_is_unchanged():
    # P29.6 is additive: the uncorrected P29.2 arm and its pin are untouched.
    assert score_train()["cl"].geometric_mean_fold_error == pytest.approx(
        2.854830092083889
    )


def test_train_mean_baseline_on_the_same_15_drugs_is_pinned():
    # The reviewer's corrected figure: the LOO geometric-mean-of-train floor
    # over exactly the 15 CL-scored drugs (not 2.751, the stale value).
    gmfe, n = cl_train_mean_baseline()
    assert n == 15
    assert gmfe == pytest.approx(2.816803165361739)


def test_fu_mic_arm_beats_the_mean_baseline_on_train():
    report = score_train_fu_mic()
    assert report["cl"].n == 15
    assert report["cl"].geometric_mean_fold_error == pytest.approx(
        1.7050282206874383
    )
    assert report["cl"].within_2_fold == pytest.approx(0.8)
    assert report["cl"].within_3_fold == pytest.approx(0.8)
    baseline, _ = cl_train_mean_baseline()
    # ACCEPTANCE 4: the arm must now be below the train-mean floor.
    assert report["cl"].geometric_mean_fold_error < baseline


def test_lipophilic_bases_improve_before_vs_after():
    before = {p.drug: p.fold_error for p in score_train()["cl"].predictions}
    after = {p.drug: p.fold_error for p in score_train_fu_mic()["cl"].predictions}
    for drug, lo, hi in [
        ("haloperidol", 4.0, 4.3),
        ("desipramine", 4.5, 4.7),
        ("chlorpromazine", 1.2, 1.3),
    ]:
        assert after[drug] < before[drug], drug
        assert lo < after[drug] < hi, drug


def test_fu_mic_predictions_carry_physchem_citations():
    for p in score_train_fu_mic()["cl"].predictions:
        if "fu_mic" in p.method:
            # logP source (ChEMBL) must appear in the input references.
            assert any("ChEMBL" in r for r in p.input_references), p.drug
