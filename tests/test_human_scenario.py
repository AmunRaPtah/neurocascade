"""The human whole-brain scenario composer: one receptor system on the
human connectome + human expression, wired into the engine.

Offline, off the committed twelve-region connectome and expression fixtures
in `src/neurocascade/human/fixtures/`. The receptor system under test is the
existing illustrative mu-opioid one (`scenarios.mu_opioid_receptor_site()` /
`scenarios.mu_opioid_pk_models()`), which is self-contained and needs no
cache. The mechanism gain is the scenarios' own illustrative
`MU_OPIOID_MECHANISM_GAIN`, which is exactly the number the composer must
NOT invent: the tests assert it is recorded as caller-supplied rather than
absorbed silently.
"""

import numpy as np
import pytest

from neurocascade import human, scenarios
from neurocascade.human import fixtures


@pytest.fixture(scope="module")
def scenario():
    return human.whole_brain_simulation(
        fixtures.load_fixture_expression(),
        "OPRM1",
        pk_models=scenarios.mu_opioid_pk_models(),
        receptor_site=scenarios.mu_opioid_receptor_site(),
        mechanism_gain=scenarios.MU_OPIOID_MECHANISM_GAIN,
        connectome=fixtures.load_fixture_connectome(),
    )


def test_scenario_builds_a_simulation_over_the_fixture_network(scenario):
    assert scenario.simulation.network.n == 12
    assert scenario.network.n == 12
    # Every node is one Desikan-Killiany region, so the coupling keys are the
    # region labels themselves and the engine resolves every one of them.
    assert set(scenario.coupling) == set(fixtures.load_fixture_connectome().labels)
    assert len(scenario.coupling) == 12


def test_the_coupling_is_negative_and_striatum_dominant(scenario):
    # mu-opioid is Gi-coupled: the mechanism gain is negative, so an agonist
    # suppresses the nodes carrying the receptor. The fixture's OPRM1 row is
    # accumbens-dominant, so that node carries the most-negative coupling.
    assert all(value < 0.0 for value in scenario.coupling.values())
    peak = min(scenario.coupling, key=scenario.coupling.get)
    assert peak == "Laccumb"


def test_the_report_records_provenance_per_layer(scenario):
    report = scenario.report
    assert report.symbol == "OPRM1"
    assert report.site_name == "mu_opioid_on_VTA_GABA"
    assert report.ligand_names == ("fentanyl", "naloxone")
    assert report.n_pk_models == 2
    assert report.connectome_key == "enigma_hcp_sc_matrix_aparc_with_sctx"
    assert report.n_nodes == 12
    assert report.n_cortical == 5
    assert report.n_subcortical == 7
    assert report.operating_gain == pytest.approx(0.9)
    assert report.normalization == "max"
    assert report.missing == "exclude"
    # The fixture has no unmeasured regions, so nothing was excluded.
    assert report.excluded_regions == ()


def test_the_report_flags_the_mechanism_gain_as_unfitted(scenario):
    assert scenario.report.mechanism_gain == pytest.approx(
        scenarios.MU_OPIOID_MECHANISM_GAIN
    )
    assert "no human mechanism_gain has ever been fitted" in (
        scenario.report.mechanism_gain_note
    )
    assert "caller-supplied" in scenario.report.mechanism_gain_note


def test_mechanism_gain_is_required():
    with pytest.raises(TypeError):
        human.whole_brain_simulation(
            fixtures.load_fixture_expression(),
            "OPRM1",
            pk_models=scenarios.mu_opioid_pk_models(),
            receptor_site=scenarios.mu_opioid_receptor_site(),
            connectome=fixtures.load_fixture_connectome(),
        )


def test_pk_model_count_must_match_the_site():
    with pytest.raises(ValueError, match="one PK model per ligand"):
        human.whole_brain_simulation(
            fixtures.load_fixture_expression(),
            "OPRM1",
            pk_models=scenarios.mu_opioid_pk_models()[:1],
            receptor_site=scenarios.mu_opioid_receptor_site(),
            mechanism_gain=scenarios.MU_OPIOID_MECHANISM_GAIN,
            connectome=fixtures.load_fixture_connectome(),
        )


def test_baseline_rests_at_the_stabilized_operating_point(scenario):
    t_eval = np.linspace(0.0, 40.0, 100)
    result = scenario.simulation.run(
        dose_concentrations=[0.0, 0.0], t_span=(0.0, 40.0), t_eval=t_eval
    )
    assert result["rates"][:, -1] == pytest.approx(0.5, abs=1e-6)


def test_a_mu_agonist_suppresses_the_brain_and_accumbens_most(scenario):
    t_eval = np.linspace(0.0, 60.0, 200)
    baseline = scenario.simulation.run(
        dose_concentrations=[0.0, 0.0], t_span=(0.0, 60.0), t_eval=t_eval
    )
    # 100 nM fentanyl is ~74x the calibrated Ki (1.346 nM, Volpe 2011), so it
    # is near-saturating; the dose magnitude is illustrative, the pattern is not.
    drug = scenario.simulation.run(
        dose_concentrations=[100.0, 0.0], t_span=(0.0, 60.0), t_eval=t_eval
    )
    delta = drug["rates"][:, -1] - baseline["rates"][:, -1]
    # Every node carrying OPRM1 is suppressed (negative gain, agonist).
    assert np.all(delta < 0.0)
    # The largest mover is the accumbens, OPRM1's peak expression region.
    labels = fixtures.load_fixture_connectome().labels
    assert labels[int(np.argmax(np.abs(delta)))] == "Laccumb"
