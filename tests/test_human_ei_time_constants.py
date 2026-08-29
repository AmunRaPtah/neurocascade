"""P13.4: measured human E/I constants remain distinguishable from fallback."""

import math
import pytest

from neurocascade.human.ei_time_constants import from_delay_maps


def test_finite_delay_values_are_measured_and_missing_values_refuse_fallback():
    constants = from_delay_maps(
        ["A", "B"], {"A": 4.0, "B": float("nan")}, {"A": 7.0},
        citation="F-TRACT", units="release units", fallback_tau=1.0,
    )
    assert constants.value("A", "excitatory") == 4.0
    assert constants.measured("A", "excitatory")
    assert constants.coverage("excitatory")["fraction"] == 0.5
    with pytest.raises(ValueError, match="no measured human excitatory"):
        constants.value("B", "excitatory")
    assert constants.value("B", "excitatory", allow_fallback=True) == 1.0


def test_unknown_labels_and_kinds_are_rejected():
    constants = from_delay_maps(["A"], {"A": 1.0}, {"A": 2.0}, citation="x", units="y")
    with pytest.raises(KeyError):
        constants.value("B", "excitatory")
    with pytest.raises(ValueError, match="kind"):
        constants.value("A", "other")
