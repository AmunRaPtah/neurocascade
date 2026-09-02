from neurocascade.human.pk_validation import known_failure_mode, validation_outcomes


def test_every_human_validation_state_is_explicit_about_what_is_unvalidated():
    outcomes = validation_outcomes()
    assert {item.drug for item in outcomes} == {"fentanyl", "haloperidol", "diazepam"}
    # No outcome carries metrics yet: nothing here has been compared to a
    # measured human observation, which is what validation would mean.
    assert all(item.metrics == () for item in outcomes)
    assert all(item.limitations for item in outcomes)


def test_only_haloperidol_has_moved_off_blocked_and_only_partly():
    blocked = {item.drug for item in validation_outcomes() if item.status.startswith("blocked-")}
    assert blocked == {"fentanyl", "diazepam"}
    haloperidol = validation_outcomes("haloperidol")[0]
    assert haloperidol.status == "partial-oral-apparent-set-transcribed"
    # The partial status has to name what is still missing, or it is just a
    # green light with extra words.
    assert "no human" in haloperidol.limitations
    assert "F is unidentified" in haloperidol.limitations


def test_haloperidol_failure_mode_is_not_hidden():
    failure = known_failure_mode("haloperidol_rat_hybrid_pbpk_pd")
    assert failure["citation"] == "Johnson 2016, PMID 26718955"
    assert "under-predicted" in failure["outcome"]
    assert "back-oxidation" in failure["cause"]
