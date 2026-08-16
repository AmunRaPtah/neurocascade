import numpy as np

from .circuit import sigmoid


class CircuitNetwork(object):
    """A set of CircuitNodes coupled by a connectivity matrix.

    connectivity[i, j] is the weight of the projection from node j onto node
    i (positive = excitatory, negative = inhibitory). Each node's rate obeys
    a Wilson-Cowan-style relaxation toward a sigmoid of its total input:

        tau_i * dr_i/dt = -r_i + sigmoid(sum_j W[i,j] * r_j + drive_i)
    """

    def __init__(self, nodes, connectivity):
        self.nodes = list(nodes)
        self.n = len(self.nodes)
        self.connectivity = np.asarray(connectivity, dtype=float)
        if self.connectivity.shape != (self.n, self.n):
            raise ValueError(
                "connectivity must be %d x %d, got %s"
                % (self.n, self.n, self.connectivity.shape)
            )
        self.index = {node.name: i for i, node in enumerate(self.nodes)}

    def baseline_drive(self):
        return np.array([node.baseline_input for node in self.nodes])

    def rhs(self, rates, drive, connectivity=None):
        """Return the vectorized Wilson-Cowan RHS.

        ``connectivity`` is an optional override for this evaluation only
        (state-dependent connection strength, e.g. an allosteric modulator
        scaling one synapse). Node parameters are broadcast as arrays so the
        per-node sigmoid loop stays out of the ODE solver's hot path.
        """
        rates = np.asarray(rates, dtype=float)
        W = self.connectivity if connectivity is None else connectivity
        net_input = W @ rates + drive
        gains = np.array([node.gain for node in self.nodes], dtype=float)
        thresholds = np.array([node.threshold for node in self.nodes], dtype=float)
        taus = np.array([node.tau for node in self.nodes], dtype=float)
        targets = sigmoid(net_input, gain=gains, threshold=thresholds)
        return (-rates + targets) / taus

    def rhs_reference(self, rates, drive, connectivity=None):
        """Reference RHS retaining the original per-node implementation.

        This is used as a numerical regression oracle; solvers use ``rhs``.
        """
        rates = np.asarray(rates, dtype=float)
        W = self.connectivity if connectivity is None else connectivity
        net_input = W @ rates + drive
        d_rates = np.empty(self.n)
        for i, node in enumerate(self.nodes):
            target = sigmoid(net_input[i], gain=node.gain, threshold=node.threshold)
            d_rates[i] = (-rates[i] + target) / node.tau
        return d_rates
