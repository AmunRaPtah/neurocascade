class OneCompartmentPK:
    """First-order elimination from a single compartment: dC/dt = -ke * C.

    Dosing is expressed as an initial concentration (bolus) rather than an
    input term, since v0 only models single-bolus exposure.
    """

    def __init__(self, ke):
        self.ke = ke

    def rhs(self, C):
        return -self.ke * C
