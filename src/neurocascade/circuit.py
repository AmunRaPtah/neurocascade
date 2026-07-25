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
