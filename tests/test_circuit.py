import numpy as np

from neurocascade import CircuitNetwork, CircuitNode, sigmoid


def test_sigmoid_bounds_and_midpoint():
    assert sigmoid(-1000) < 1e-9
    assert sigmoid(1000) > 1.0 - 1e-9
    assert abs(sigmoid(0.0) - 0.5) < 1e-9


def test_two_node_network_relaxes_toward_sigmoid_fixed_point():
    a = CircuitNode(name="A", tau=1.0, baseline_input=2.0)
    b = CircuitNode(name="B", tau=1.0, baseline_input=0.0)
    W = [[0.0, 0.0], [1.0, 0.0]]  # B receives from A, A receives nothing
    net = CircuitNetwork(nodes=[a, b], connectivity=W)

    rates = np.array([0.0, 0.0])
    drive = net.baseline_drive()
    dt = 0.01
    for _ in range(200000):
        rates = rates + dt * net.rhs(rates, drive)

    expected_a = sigmoid(2.0)
    assert abs(rates[0] - expected_a) < 1e-3
    expected_b = sigmoid(rates[0])
    assert abs(rates[1] - expected_b) < 1e-3


def test_inhibitory_connection_suppresses_downstream_node():
    a = CircuitNode(name="A", tau=1.0, baseline_input=5.0)
    b = CircuitNode(name="B", tau=1.0, baseline_input=1.0)
    W_excitatory = [[0.0, 0.0], [1.0, 0.0]]
    W_inhibitory = [[0.0, 0.0], [-1.0, 0.0]]

    def steady_state(W):
        net = CircuitNetwork(nodes=[a, b], connectivity=W)
        rates = np.array([0.0, 0.0])
        drive = net.baseline_drive()
        dt = 0.01
        for _ in range(200000):
            rates = rates + dt * net.rhs(rates, drive)
        return rates

    excitatory_b = steady_state(W_excitatory)[1]
    inhibitory_b = steady_state(W_inhibitory)[1]
    assert inhibitory_b < excitatory_b
