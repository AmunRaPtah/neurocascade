import numpy as np


class Ligand:
    """A drug competing for a receptor site.

    efficacy: 1.0 = full agonist, 0.0 = neutral antagonist (occupies but does
    not activate), fractional = partial agonist, negative = inverse agonist.
    kon/koff set the association/dissociation rate constants; affinity is
    koff/kon (lower = tighter binding).
    """

    def __init__(self, name, kon, koff, efficacy):
        self.name = name
        self.kon = kon
        self.koff = koff
        self.efficacy = efficacy


class ReceptorSite:
    """A receptor population competitively bound by one or more ligands.

    State is the fractional occupancy owned by each ligand (they share one
    finite pool of receptor, so occupancies sum to at most 1). This is what
    lets an antagonist (e.g. naloxone) displace an agonist (e.g. fentanyl)
    already bound at the same site.
    """

    def __init__(self, name, ligands):
        self.name = name
        self.ligands = list(ligands)
        self.n = len(self.ligands)

    def occupancy_rhs(self, occupancy, concentrations):
        occupancy = np.asarray(occupancy, dtype=float)
        concentrations = np.asarray(concentrations, dtype=float)
        free_fraction = 1.0 - occupancy.sum()
        d_occ = np.empty(self.n)
        for i, lig in enumerate(self.ligands):
            d_occ[i] = lig.kon * concentrations[i] * free_fraction - lig.koff * occupancy[i]
        return d_occ

    def activation(self, occupancy):
        return sum(o * lig.efficacy for o, lig in zip(occupancy, self.ligands))
