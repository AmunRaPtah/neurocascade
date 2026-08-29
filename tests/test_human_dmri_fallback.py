"""P13.6: dMRI is usable only as an inferred fallback."""

import numpy as np
import pytest

from neurocascade.human import fixtures
from neurocascade.human.dmri_fallback import DMRIFallback
from neurocascade.human.ftract import FTractFeature
from neurocascade.human.provenance import INFERRED


def test_dmri_fallback_never_passes_measured_requirement():
    fallback = DMRIFallback(fixtures.load_fixture_connectome())
    source, target = fallback.labels[0], fallback.labels[1]
    if fallback.connectome.weights[0, 1] == 0:
        source, target = next((fallback.labels[i], fallback.labels[j]) for i, j in zip(*np.where(fallback.connectome.weights != 0)))
    assert fallback.require_inferred(source, target) == INFERRED
    with pytest.raises(ValueError, match="never measured"):
        fallback.require_measured(source, target)


def test_dmri_and_ftract_different_parcellations_are_not_compared():
    fallback = DMRIFallback(fixtures.load_fixture_connectome())
    feature = FTractFeature("probability", ["x", "y"], np.zeros((2, 2)))
    result = fallback.compare_with_ftract(feature)
    assert result["status"] == "not_comparable"
    assert result["n_compared"] == 0
    assert "parcellation" in result["reason"]
