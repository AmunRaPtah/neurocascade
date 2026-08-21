import numpy as np


def sigmoid(x, gain=1.0, threshold=0.0):
    z = np.clip(gain * (x - threshold), -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


class CircuitNode(object):
    """A single neural population (rate/mass-model unit).

    tau: time constant of the population's response.
    baseline_input: tonic drive present with no drug on board.
    gain/threshold: shape of the population's input-output sigmoid.
    """

    def __init__(self, name, tau, baseline_input=0.0, gain=1.0, threshold=0.0):
        self.name = name
        self.tau = tau
        self.baseline_input = baseline_input
        self.gain = gain
        self.threshold = threshold


#: Illustrative default node parameters for a connectome-built network.
#:
#: Hoisted out of `connectome/build.py` in P11.1: both the mouse and the human
#: builders need them, and neither should have to import the other's package to
#: get them. These values are ILLUSTRATIVE and always have been. P13.4 replaces
#: `tau` and the E/I balance with F-TRACT-measured human synaptic time
#: constants, which will be the first circuit-layer parameters in this
#: project's history to have a measured human source.
DEFAULT_NODE = {"tau": 1.0, "baseline_input": 0.0, "gain": 4.0, "threshold": 0.0}
