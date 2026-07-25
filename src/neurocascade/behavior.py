class BehaviorReadout(object):
    """A named behavioral axis decoded as a linear readout of circuit rates.

    weights: dict mapping node name -> weight. Nodes not present in the
    weights dict don't contribute. This is deliberately the simplest
    possible decoder (interpretable, no fitting required) so v0 can be
    checked by hand; a learned/nonlinear decoder is a natural upgrade once
    there's behavioral endpoint data to fit against.
    """

    def __init__(self, name, weights):
        self.name = name
        self.weights = dict(weights)

    def score(self, network, rates):
        total = 0.0
        for node_name, weight in self.weights.items():
            idx = network.index[node_name]
            total += weight * rates[idx]
        return total
