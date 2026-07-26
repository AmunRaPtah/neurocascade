import numpy as np
from scipy.integrate import solve_ivp


class Simulation(object):
    """Wires pharmacokinetics -> receptor binding -> circuit network into one
    coupled ODE system and integrates it with RK45 (as in the CCT model).

    State vector layout: [C_1..C_d, occ_1..occ_d, r_1..r_n]
      C   : plasma concentration per ligand (len == receptor_site.n)
      occ : fractional receptor occupancy per ligand (same order as C)
      r   : firing rate per circuit node (len == network.n)

    coupling: dict of node_name -> gain. Receptor activation (in [-1, 1],
    signed by each ligand's efficacy) is added to that node's drive scaled
    by gain. A negative gain models an inhibitory receptor (e.g. Gi-coupled
    mu-opioid receptors on GABAergic interneurons): agonist activation
    *reduces* that node's input.

    connection_modulation: optional dict of (post_node_name, pre_node_name)
    -> gain, for receptors that don't drive a node directly but instead
    modulate the strength of an existing connection -- an allosteric
    modulator (e.g. a benzodiazepine at GABA-A) that potentiates or dampens
    an already-present synapse rather than gating current itself. The
    targeted connectivity entry is scaled by `1 + gain * activation` each
    step; everything else in the network is unaffected. This is additive to,
    not a replacement for, `coupling` -- a receptor can use either, both, or
    neither depending on its mechanism.
    """

    def __init__(self, pk_models, receptor_site, network, coupling, connection_modulation=None):
        if len(pk_models) != receptor_site.n:
            raise ValueError("need one PK model per ligand on the receptor site")
        self.pk_models = list(pk_models)
        self.receptor_site = receptor_site
        self.network = network
        self.coupling = dict(coupling)
        self.connection_modulation = dict(connection_modulation or {})
        self.d = len(self.pk_models)
        self.r = receptor_site.n
        self.c = network.n

    def _unpack(self, y):
        d, r = self.d, self.r
        C = y[0:d]
        occ = y[d : d + r]
        rates = y[d + r : d + r + self.c]
        return C, occ, rates

    def rhs(self, t, y):
        C, occ, rates = self._unpack(y)

        dC = np.array([pk.rhs(ci) for pk, ci in zip(self.pk_models, C)])
        d_occ = self.receptor_site.occupancy_rhs(occ, C)

        activation = self.receptor_site.activation(occ)
        drive = self.network.baseline_drive()
        for node_name, gain in self.coupling.items():
            idx = self.network.index[node_name]
            drive[idx] += gain * activation

        connectivity = None
        if self.connection_modulation:
            connectivity = self.network.connectivity.copy()
            for (post_name, pre_name), gain in self.connection_modulation.items():
                i = self.network.index[post_name]
                j = self.network.index[pre_name]
                connectivity[i, j] = connectivity[i, j] * (1.0 + gain * activation)
        d_rates = self.network.rhs(rates, drive, connectivity=connectivity)

        return np.concatenate([dC, d_occ, d_rates])

    def run(self, dose_concentrations, t_span, t_eval=None, rates0=None):
        """dose_concentrations: initial plasma concentration per ligand
        (bolus at t=0), in the same order as receptor_site.ligands."""
        C0 = np.asarray(dose_concentrations, dtype=float)
        if C0.shape != (self.r,):
            raise ValueError("dose_concentrations must have length %d" % self.r)
        occ0 = np.zeros(self.r)
        if rates0 is None:
            rates0 = np.zeros(self.c)
        y0 = np.concatenate([C0, occ0, np.asarray(rates0, dtype=float)])

        sol = solve_ivp(
            self.rhs, t_span, y0, method="RK45", t_eval=t_eval, max_step=0.1
        )
        if not sol.success:
            raise RuntimeError("integration failed: %s" % sol.message)

        d, r = self.d, self.r
        return {
            "t": sol.t,
            "concentrations": sol.y[0:d],
            "occupancy": sol.y[d : d + r],
            "rates": sol.y[d + r : d + r + self.c],
        }
