"""The FDA population-PK bridge: what it converts, what it refuses, and whether
the conversion reproduces the review's own simulated exposures.

The last of those is the only test here that could have failed for an interesting
reason. The FDA multidisciplinary review for LYBALVI prints, next to the popPK
parameter table, the steady-state exposures its authors simulated for the
reference subject. Those are not independent validation -- they come out of the
same fit -- but they are a sharp check that the bridge put every number in the
right slot and in the right unit, because a transposed volume or an hour/minute
slip moves them by far more than the few percent seen here.
"""

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from neurocascade.human import fda_popk
from neurocascade.human.fda_pk_bridge import (
    bridge_survey,
    bridgeable,
    bridged_records,
    dropped_parameters,
    refusals,
    to_human_pk_record,
)
from neurocascade.human.pk_parameters import human_pk_records
from neurocascade.pk import DosedTwoCompartmentPK

#: Olanzapine free base, C17H20N4S. Used only to turn the model's nM back into
#: the ng/mL the FDA review prints.
OLANZAPINE_G_PER_MOL = 312.43


def _lybalvi():
    matches = [
        record
        for record in fda_popk.FDA_POPPK_RECORDS
        if record.application_number == "NDA 213378"
    ]
    assert len(matches) == 1, "expected exactly one LYBALVI row, got %d" % len(matches)
    return matches[0]


def test_exactly_the_rows_that_convert_convert():
    """The bridge is narrow, and the test states how narrow rather than asserting
    a floor. If a later phase transcribes a new FDA row this test fails and the
    new row has to be classified deliberately."""
    converted = [
        record.application_number + " " + record.route + " " + record.analyte
        for record in fda_popk.FDA_POPPK_RECORDS
        if bridgeable(record)
    ]
    assert converted == ["NDA 213378 PO parent"]


def test_every_refusal_names_a_property_of_the_source():
    """No row may be refused with an empty or generic reason: the point of the
    refusal is that the reader learns what is wrong with the source."""
    for record in fda_popk.FDA_POPPK_RECORDS:
        for reason in refusals(record):
            assert reason.strip()
            assert len(reason) > 30, (record.application_number, reason)


def test_refusal_reasons_are_the_expected_kinds():
    by_application = {}
    for record in fda_popk.FDA_POPPK_RECORDS:
        by_application.setdefault(record.application_number, []).extend(
            refusals(record)
        )
    # Parent-and-metabolite fits are refused, not collapsed onto the parent.
    assert any(
        "analyte is" in reason
        for reason in by_application["NDA 210655"]
    )
    # A depot disposition model is not reused for an oral prediction.
    assert any(
        "oral only" in reason
        for record in fda_popk.FDA_POPPK_RECORDS
        if record.route != "PO"
        for reason in refusals(record)
    )


def test_searched_and_empty_rows_are_refused_with_their_barrier():
    for record in fda_popk.FDA_POPPK_RECORDS:
        if record.status == "transcribed":
            continue
        reasons = refusals(record)
        assert reasons, record.application_number
        assert record.extraction_barrier in reasons[0]


def test_to_human_pk_record_raises_rather_than_returning_none():
    unconvertible = next(
        record for record in fda_popk.FDA_POPPK_RECORDS if not bridgeable(record)
    )
    with pytest.raises(ValueError) as excinfo:
        to_human_pk_record(unconvertible)
    assert unconvertible.application_number in str(excinfo.value)


def test_bridged_record_carries_the_apparent_flag_and_no_bioavailability():
    bridged = to_human_pk_record(_lybalvi())
    assert bridged.apparent_over_F is True
    assert bridged.units == "absolute"
    assert bridged.status == "fda-population-pk-bridged"
    assert bridged.reference_weight_kg == 70.0
    assert bridged.n_compartments == 2
    # The review never identified F, so the record must not claim one.
    assert getattr(bridged, "F", None) in (None, )


def test_bridged_values_match_the_printed_table():
    bridged = to_human_pk_record(_lybalvi())
    assert bridged.CL_L_per_h == pytest.approx(15.5)
    assert bridged.V1_L == pytest.approx(656.0)
    assert bridged.V2_L == pytest.approx(225.0)
    assert bridged.Q12_L_per_h == pytest.approx(6.15)
    assert bridged.ka_per_min == pytest.approx(0.861 / 60.0)
    assert bridged.cl_allometric_exponent == pytest.approx(0.75)


def test_the_lag_time_is_dropped_and_says_why():
    dropped = dict(dropped_parameters(_lybalvi()))
    assert "ALAG" in dropped
    assert "steady state" in dropped["ALAG"]


def test_bridged_records_are_reachable_through_the_ordinary_registry():
    """A bridged row has to be visible to the same lookup that finds a
    hand-transcribed one, or nothing downstream can use it."""
    olanzapine = human_pk_records("olanzapine", "PO")
    assert len(olanzapine) == 1
    assert olanzapine[0].status == "fda-population-pk-bridged"


def test_hand_transcribed_rows_come_first_and_are_unchanged():
    """The append must not reorder or shadow anything typed by hand."""
    from neurocascade.human.pk_parameters import HUMAN_PK_RECORDS

    combined = human_pk_records()
    assert combined[: len(HUMAN_PK_RECORDS)] == tuple(HUMAN_PK_RECORDS)
    assert len(combined) == len(HUMAN_PK_RECORDS) + len(bridged_records())


def test_bridge_survey_covers_every_row():
    lines = bridge_survey().splitlines()
    assert len(lines) == len(fda_popk.FDA_POPPK_RECORDS)
    assert sum(line.endswith("BRIDGED") for line in lines) == 1


def _steady_state_ng_per_ml(dose_mg, n_doses=40, interval_min=1440.0, step_min=5.0):
    """Concentration over the last dosing interval, by superposition.

    The PK is linear, so summing the single-dose curve at one-interval offsets is
    exact rather than approximate; this is the same construction
    occupancy_validation uses, kept here in the PK units the FDA review prints.
    """
    record = human_pk_records("olanzapine", "PO")[0]
    weight = record.reference_weight_kg
    model = DosedTwoCompartmentPK(
        route="PO",
        V1_L_per_kg=record.V1_L / weight,
        V2_L_per_kg=record.V2_L / weight,
        Q_mL_per_min_per_kg=record.Q12_L_per_h * 1000.0 / 60.0 / weight,
        CL_mL_per_min_per_kg=record.CL_L_per_h * 1000.0 / 60.0 / weight,
        body_weight_kg=weight,
        ka_per_min=record.ka_per_min,
        molar_mass_g_per_mol=OLANZAPINE_G_PER_MOL,
        apparent_parameters=True,
        species="human",
    )
    times = np.arange(0.0, n_doses * interval_min + step_min, step_min)
    solution = solve_ivp(
        lambda t, state: model.concentration_rhs(state),
        (0.0, times[-1]),
        model.initial_state(dose_mg / weight),
        t_eval=times,
        method="LSODA",
        rtol=1e-10,
        atol=1e-14,
    )
    nM = np.clip(model.receptor_concentration(solution.y), 0.0, None)
    per_interval = int(interval_min / step_min)
    n_full = len(nM) // per_interval
    accumulated = nM[: n_full * per_interval].reshape(n_full, per_interval).sum(axis=0)
    return accumulated * OLANZAPINE_G_PER_MOL / 1000.0


def test_bridged_record_reproduces_the_reviews_own_steady_state_exposures():
    """LYBALVI review Figures 15 and 16, PDF pp.286-287, print the exposures the
    sponsor simulated for the reference subject at 10 mg once daily:
    Cmax,ss = 31.7 ng/mL and AUCtau = 635 ng.h/mL.

    These come out of the same fit as the parameters, so agreement is not
    independent evidence that the model is right about olanzapine. It is evidence
    that the bridge is right about the model. The residual few percent is the
    difference between a full popPK simulation over a covariate distribution and
    a single typical subject with the absorption lag dropped, which is the
    documented scope of the conversion.
    """
    ng_per_ml = _steady_state_ng_per_ml(10.0)
    auc_tau = ng_per_ml.mean() * 24.0
    assert auc_tau == pytest.approx(635.0, rel=0.05)
    assert ng_per_ml.max() == pytest.approx(31.7, rel=0.05)


def test_steady_state_auc_equals_dose_over_apparent_clearance():
    """The exact analytic identity for any linear model: AUC over one interval at
    steady state is Dose/(CL/F), whatever the distribution structure. This is the
    test that would catch a volume in the clearance slot."""
    record = human_pk_records("olanzapine", "PO")[0]
    ng_per_ml = _steady_state_ng_per_ml(10.0)
    analytic = 10.0 / record.CL_L_per_h * 1000.0
    assert ng_per_ml.mean() * 24.0 == pytest.approx(analytic, rel=1e-3)
