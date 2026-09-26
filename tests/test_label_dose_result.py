"""P22.2 result: predicted windows scored against transcribed FDA label ranges."""
import hashlib
import math
from pathlib import Path

import pytest

from neurocascade.human import fda_sources
from neurocascade.human.label_dose_prediction import DENOMINATOR
from neurocascade.human.label_dose_result import (
    LABEL_TRANSCRIPTIONS,
    LabelDoseRange,
    LabelReading,
    label_dose_summary,
    label_dose_table,
)

FDA_DIR = Path(__file__).resolve().parent.parent / "data" / "fda"


@pytest.fixture(scope="module")
def table():
    return {row.drug: row for row in label_dose_table()}


def test_every_denominator_drug_appears_once_in_order():
    assert tuple(row.drug for row in label_dose_table()) == DENOMINATOR


def test_only_the_drivable_drugs_have_a_transcribed_label(table):
    # Refused drugs are kept untranscribed so a later prediction for them can
    # still precede its ground truth (deviation D2).
    assert set(LABEL_TRANSCRIPTIONS) == {"haloperidol", "olanzapine"}
    for drug, row in table.items():
        if drug in LABEL_TRANSCRIPTIONS:
            assert row.label is LABEL_TRANSCRIPTIONS[drug]
            assert row.score.refusal is None
        else:
            assert row.label is None
            assert row.score.refusal == row.window.refused_layers[0]
            assert row.score.passed is False and row.score.e is None


def test_refusal_attribution_is_pinned(table):
    assert {d: table[d].score.refusal for d in DENOMINATOR if table[d].label is None} == {
        "risperidone": "active-metabolite",
        "paliperidone": "pk",
        "quetiapine": "pk",
        "ziprasidone": "pk",
        "aripiprazole": "active-metabolite",
        "clozapine": "pk",
    }


def test_label_ranges_as_transcribed():
    hal = LABEL_TRANSCRIPTIONS["haloperidol"]
    assert [(r.l_low, r.l_high) for r in hal.readings] == [(1.0, 15.0), (1.0, 100.0)]
    assert (hal.scoring_reading.l_low, hal.scoring_reading.l_high) == (1.0, 15.0)
    olz = LABEL_TRANSCRIPTIONS["olanzapine"]
    assert [(r.l_low, r.l_high) for r in olz.readings] == [(10.0, 20.0)]


def test_haloperidol_scores(table):
    row = table["haloperidol"]
    assert row.window.blind is False
    assert row.score.overlap and row.score.passed
    expected = math.log10(math.sqrt(2.89 * 5.56) / math.sqrt(1.0 * 15.0))
    assert row.score.e == pytest.approx(expected, abs=1e-12)
    assert row.score.e == pytest.approx(0.0149407, abs=1e-7)
    ((name, other),) = row.other_readings
    assert name == "up to 100 mg"
    assert other.overlap and not other.passed
    assert other.e == pytest.approx(-0.3970137, abs=1e-7)


def test_olanzapine_scores(table):
    row = table["olanzapine"]
    assert row.window.blind is True
    assert row.score.overlap and row.score.passed
    expected = math.log10(math.sqrt(5.77 * 16.4) / math.sqrt(10.0 * 20.0))
    assert row.score.e == pytest.approx(expected, abs=1e-12)
    assert row.score.e == pytest.approx(-0.1625052, abs=1e-7)
    assert row.other_readings == ()


def test_summary(table):
    summary = label_dose_summary(tuple(table[d] for d in DENOMINATOR))
    assert summary["n_pass"] == 2
    assert summary["n_scored"] == 2
    assert summary["denominator"] == 8
    assert summary["max_abs_e"] == pytest.approx(0.1625052, abs=1e-7)
    assert summary["median_abs_e"] == pytest.approx((0.0149407 + 0.1625052) / 2, abs=1e-6)


def test_occupancy_profile_brackets_the_mean(table):
    for drug in LABEL_TRANSCRIPTIONS:
        profile = table[drug].occupancy_profile
        assert [p[0] for p in profile] == [0.65, 0.78]
        for target, _dose, trough, mean, peak in profile:
            assert trough < mean < peak
            assert mean == pytest.approx(target, abs=1e-4)


def test_provenance_agrees_with_the_p20_6_record():
    for drug, label in LABEL_TRANSCRIPTIONS.items():
        record = fda_sources.LABELS_BY_DRUG[drug]
        assert label.application_number in record["application_number"]
        assert label.effective_time == record["effective_time"]
        assert label.spl_set_id == record["spl_set_id"]
        assert label.matches_p20_6_pin == (label.sha256_read == record["sha256"])
    assert LABEL_TRANSCRIPTIONS["haloperidol"].matches_p20_6_pin is False
    assert LABEL_TRANSCRIPTIONS["olanzapine"].matches_p20_6_pin is True


def test_readings_are_validated():
    with pytest.raises(ValueError):
        LabelDoseRange("x", "NDA0", "0", "0", "0", True, (), ())
    with pytest.raises(ValueError):
        LabelDoseRange("x", "NDA0", "0", "0", "0", True, (), (LabelReading("r", 5.0, 1.0, ""),))


@pytest.mark.skipif(
    not FDA_DIR.is_dir(), reason="data/fda/ is gitignored; run scripts/fetch_fda_labels.py"
)
@pytest.mark.parametrize("drug", sorted(LABEL_TRANSCRIPTIONS))
def test_quotes_are_verbatim_in_the_hashed_snapshot(drug):
    label = LABEL_TRANSCRIPTIONS[drug]
    payload = (FDA_DIR / ("label-%s.json" % drug)).read_bytes()
    assert hashlib.sha256(payload).hexdigest() == label.sha256_read
    dosage = (FDA_DIR / ("dosage-%s.txt" % drug)).read_text()
    for quote in label.quotes:
        assert quote in dosage
