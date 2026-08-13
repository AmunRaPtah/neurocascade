"""Nicotinic desensitization and muscarinic block on one circuit (P5.4).

Prints six conditions end to end (PK -> receptor state -> circuit rates ->
readouts) for a basal-forebrain/cortex network carrying four acetylcholine
receptor systems: alpha4beta2, alpha7 and M1 on one cortical pyramidal
node, and the M2 autoreceptor on the cholinergic source nucleus.

What this example exists to show is the thing no scenario before it could
show, because no receptor before it had a third state:

  1. acetylcholine alone: all four receptors engaged by one transmitter,
     and the clearest single picture of what the new state buys. At t = 10
     min the transmitter itself is long gone (`ACETYLCHOLINE_KE` clears it
     with a half-life of 8 seconds) and every conducting and every
     competitive occupancy has returned to zero, but 12% of the alpha4beta2
     pool is STILL desensitized, because that site's measured recovery
     constant is 86.9 min. A two-state model has nowhere to put that 12%,
     and the alpha7 pool, whose recovery constant is 1.9 min, has already
     emptied.
  2. nicotine alone: the alpha4beta2 pool ends up mostly DESENSITIZED
     rather than conducting. The ratio is k_des/k_rec = 24.5 at
     alpha4beta2 and 4.8 at alpha7, both from [Fenster1997]'s Table 1, and
     the difference between those two numbers is the whole reason the two
     nicotinic sites sit on one node with one gain.
  3. nicotine + mecamylamine: the channel blocker cuts the conducting
     fraction without touching either agonist's occupancy, which is what
     "non-competitive" costs the model to say.
  4. acetylcholine + scopolamine: the muscarinic contrast, where the
     antagonist DOES displace, because scopolamine really does compete for
     the orthosteric site.
  5. mecamylamine alone and 6. scopolamine alone: both inert, because this
     model has no endogenous cholinergic tone for an antagonist to remove.
     That is a stated limitation, not a result; P6.2 is where it changes.

Every affinity, and both gating rate constants at each nicotinic site, is
literature-sourced; the circuit, the association rate constant and the
pharmacokinetics are illustrative and labelled. See `acetylcholine.py` for
the sources, the four derivations, the muscarinic affinity-state problem,
and the 10-fold shortfall the desensitized state's own prediction has
against the measurement.

Run with the shared Pardalos venv:
    /root/projects/pardalos/.venv/bin/python examples/acetylcholine_nicotinic_muscarinic_circuit.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from neurocascade import Simulation, acetylcholine  # noqa: E402

T_END = 10.0

#: Doses are achieved concentrations in nM (`OneCompartmentPK`), not mg/kg.
#: Kept modest because the nicotinic sites carry koff values of several
#: hundred per minute and RK45's step size scales with them.
ACETYLCHOLINE_nM = 20000.0
NICOTINE_nM = 20000.0
MECAMYLAMINE_nM = 20000.0
SCOPOLAMINE_nM = 100.0


def build():
    return Simulation(
        network=acetylcholine.basal_forebrain_cortical_network(),
        receptor_systems=acetylcholine.cholinergic_receptor_systems(),
    )


def run(sim, t_eval, **doses):
    return sim.run(
        dose_concentrations=sim.dose_vector(**doses),
        t_span=(0.0, T_END),
        t_eval=t_eval,
        rates0=acetylcholine.rest_rates(sim.network),
    )


def describe(sim, label, result):
    tone = acetylcholine.cholinergic_tone_readout()
    cortex = acetylcholine.cortical_output_readout()
    rates = result["rates"][:, -1]
    print("\n%s" % label)
    print(
        "  cholinergic_tone %.4f   cortical_output %.4f"
        % (tone.score(sim.network, rates), cortex.score(sim.network, rates))
    )
    for index, system in enumerate(sim.receptor_systems):
        site = system.receptor_site
        block = result["occupancy"][sim.state_rows(index)][:, -1]
        if hasattr(site, "split_states"):
            bound, desens, blocked = site.split_states(block)
            print(
                "  %-11s conducting %s  desensitized %s  channel-blocked %.3f"
                % (
                    system.name,
                    np.array2string(bound, precision=4, suppress_small=True),
                    np.array2string(desens, precision=4, suppress_small=True),
                    blocked.sum(),
                )
            )
        else:
            print(
                "  %-11s occupancy   %s"
                % (
                    system.name,
                    np.array2string(block, precision=4, suppress_small=True),
                )
            )


def main():
    sim = build()
    t_eval = np.linspace(0.0, T_END, 201)

    print(
        "ligand order per nicotinic site: acetylcholine, nicotine "
        "(+ mecamylamine in the channel pool)"
    )
    print("ligand order per muscarinic site: acetylcholine, scopolamine")

    describe(sim, "1. acetylcholine alone", run(sim, t_eval, acetylcholine=ACETYLCHOLINE_nM))
    describe(sim, "2. nicotine alone", run(sim, t_eval, nicotine=NICOTINE_nM))
    describe(
        sim,
        "3. nicotine + mecamylamine",
        run(sim, t_eval, nicotine=NICOTINE_nM, mecamylamine=MECAMYLAMINE_nM),
    )
    describe(
        sim,
        "4. acetylcholine + scopolamine",
        run(
            sim,
            t_eval,
            acetylcholine=ACETYLCHOLINE_nM,
            scopolamine=SCOPOLAMINE_nM,
        ),
    )
    describe(sim, "5. mecamylamine alone", run(sim, t_eval, mecamylamine=MECAMYLAMINE_nM))
    describe(sim, "6. scopolamine alone", run(sim, t_eval, scopolamine=SCOPOLAMINE_nM))

    print("\ntrapping ratio k_des/k_rec, from [Fenster1997] Table 1:")
    for system in sim.receptor_systems[:2]:
        site = system.receptor_site
        print(
            "  %-11s %.2f  (apparent affinity factor %.2f)"
            % (
                system.name,
                site.k_des[0] / site.k_rec[0],
                site.apparent_affinity_factor()[0],
            )
        )


if __name__ == "__main__":
    main()
