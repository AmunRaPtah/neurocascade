from neurocascade.human.pk_validation import known_failure_mode, validation_outcomes


def test_all_three_human_validation_states_are_explicitly_blocked():
    outcomes = validation_outcomes()
    assert {item.drug for item in outcomes} == {"fentanyl", "haloperidol", "diazepam"}
    assert all(item.status.startswith("blocked-") for item in outcomes)


def test_haloperidol_failure_mode_is_not_hidden():
    failure = known_failure_mode("haloperidol_rat_hybrid_pbpk_pd")
    assert failure["citation"] == "Johnson 2016, PMID 26718955"
    assert "under-predicted" in failure["outcome"]
    assert "back-oxidation" in failure["cause"]
