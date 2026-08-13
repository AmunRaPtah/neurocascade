"""alpha1, alpha2 and beta adrenoceptors on one LC/cortex circuit (P5.3).

Runs nine conditions end to end (PK -> receptor occupancy -> circuit rates
-> readouts) and plots them: a drug-free control, the seven numbered below,
and a larger yohimbine dose. Three receptor systems drive ONE shared network
in ONE `solve_ivp` call, two of them landing on the same cortical node with
the same sign and one on the source nucleus with the opposite sign:

  1. noradrenaline alone: the whole point of the scenario. One transmitter,
     three receptors, two nodes moving in OPPOSITE directions -- cortex up
     through alpha1 (Gq) and beta1 (Gs), locus coeruleus down through the
     alpha2 (Gi) autoreceptor.
  2. noradrenaline + prazosin: the alpha1 component removed.
  3. noradrenaline + propranolol: the beta1 component removed. This is the
     LARGER of the two, and that is a prediction rather than a parameter:
     the two mechanism gains are equal by construction, and noradrenaline
     binds beta1 8.5-fold more tightly than alpha1A ([Baker2010] against
     [Proudman2021]).
  4. noradrenaline + both: nearly the whole cortical response gone, because
     the node carries no other drug target.
  5. clonidine alone: alpha2 agonist at the somatodendritic autoreceptor.
     The LC node falls -- the [Cedarbaum1977] result -- and it falls only
     0.32 as far as a full agonist would, because that is clonidine's
     measured intrinsic activity ([Jasper1998]).
  6. clonidine, then yohimbine later: displacement at the shared alpha2
     pool. Reversal is PARTIAL at an equimolar dose, because yohimbine is
     only 1.8-fold tighter than clonidine in this affinity state, and the
     example shows a 5x dose reversing further.
  7. noradrenaline + clonidine: the result this scenario predicted
     backwards. Clonidine SUPPRESSES the LC alone and RAISES it against a
     noradrenaline background, because a partial agonist competing with a
     full agonist for one pool is a functional antagonist.

Every binding constant and efficacy is literature-sourced; the circuit and
the pharmacokinetics are illustrative. See `noradrenaline.py` for the
sources, the three derivations, the 10000-fold noradrenaline affinity-state
problem, and the alpha2 autoreceptor's Phase 6 dependency.

Run with the shared Pardalos venv:
    /root/projects/pardalos/.venv/bin/python examples/noradrenaline_locus_coeruleus_cortex.py
"""

import os
import sys

import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from neurocascade import Simulation, noradrenaline  # noqa: E402

T_END = 20.0
T_SPLIT = 8.0  # when the late antagonist dose arrives

#: See `tests/test_noradrenaline_scenario.py` for why these doses and not
#: larger ones: the alpha2 site's shared kon makes RK45's step size scale
#: with the noradrenaline concentration.
NORADRENALINE_nM = 3000.0
PRAZOSIN_nM = 100.0
PROPRANOLOL_nM = 100.0
CLONIDINE_nM = 1000.0
YOHIMBINE_nM = 1000.0


def build():
    network = noradrenaline.locus_coeruleus_cortical_network()
    sim = Simulation(
        network=network,
        receptor_systems=noradrenaline.noradrenergic_receptor_systems(),
    )
    return sim


def run(sim, t_eval, **doses):
    """One condition, started from the network's rest state."""
    result = sim.run(
        dose_concentrations=sim.dose_vector(**doses),
        t_span=(0.0, T_END),
        t_eval=t_eval,
        rates0=noradrenaline.rest_rates(sim.network),
    )
    return result


def run_with_late_dose(sim, t_eval, first, second):
    """Dose once at t=0 and again at `T_SPLIT`, as a state edit.

    `Simulation.add_dose` is what a second administration physically is in a
    linear PK model: superposition onto whatever is already on board. Done
    the same way as `examples/fentanyl_reward_circuit.py`.
    """
    early = t_eval[t_eval <= T_SPLIT]
    late = t_eval[t_eval > T_SPLIT]

    y0 = np.concatenate(
        [
            sim.initial_pk_state(sim.dose_vector(**first)),
            np.zeros(sim.r),
            noradrenaline.rest_rates(sim.network),
        ]
    )
    first_leg = solve_ivp(
        sim.rhs, (0.0, T_SPLIT), y0, method="RK45", t_eval=early, max_step=0.1
    )
    if not first_leg.success:
        raise RuntimeError(first_leg.message)

    y_split = sim.add_dose(first_leg.y[:, -1], sim.dose_vector(**second))
    second_leg = solve_ivp(
        sim.rhs, (T_SPLIT, T_END), y_split, method="RK45", t_eval=late, max_step=0.1
    )
    if not second_leg.success:
        raise RuntimeError(second_leg.message)

    class _Joined(object):
        t = np.concatenate([first_leg.t, second_leg.t])
        y = np.concatenate([first_leg.y, second_leg.y], axis=1)

    return sim.unpack_solution(_Joined())


def main():
    sim = build()
    t_eval = np.linspace(0.0, T_END, 400)
    tone = noradrenaline.noradrenergic_tone_readout()
    cortex = noradrenaline.cortical_output_readout()

    conditions = []

    def record(label, result):
        rates = result["rates"]
        tone_trace = np.array(
            [tone.score(sim.network, rates[:, i]) for i in range(rates.shape[1])]
        )
        cortex_trace = np.array(
            [cortex.score(sim.network, rates[:, i]) for i in range(rates.shape[1])]
        )
        conditions.append((label, result["t"], tone_trace, cortex_trace))
        # Both the extreme and the final value, because for the late-dose
        # conditions the LC minimum happens BEFORE the antagonist arrives:
        # the reversal is only visible at the end of the trace.
        print(
            "%-42s LC min %.4f  LC end %.4f  cortex max %.4f"
            % (label, tone_trace.min(), tone_trace[-1], cortex_trace.max())
        )
        return tone_trace, cortex_trace

    print("drug-free baseline is 0.5 at every node (rule 1); "
          "every deviation below is the drug.\n")

    record("drug-free", run(sim, t_eval))
    _, cortex_na = record(
        "noradrenaline", run(sim, t_eval, noradrenaline=NORADRENALINE_nM)
    )
    _, cortex_no_alpha1 = record(
        "noradrenaline + prazosin",
        run(sim, t_eval, noradrenaline=NORADRENALINE_nM, prazosin=PRAZOSIN_nM),
    )
    _, cortex_no_beta = record(
        "noradrenaline + propranolol",
        run(sim, t_eval, noradrenaline=NORADRENALINE_nM, propranolol=PROPRANOLOL_nM),
    )
    _, cortex_neither = record(
        "noradrenaline + prazosin + propranolol",
        run(
            sim,
            t_eval,
            noradrenaline=NORADRENALINE_nM,
            prazosin=PRAZOSIN_nM,
            propranolol=PROPRANOLOL_nM,
        ),
    )
    tone_clonidine, _ = record(
        "clonidine", run(sim, t_eval, clonidine=CLONIDINE_nM)
    )
    record(
        "clonidine, then yohimbine at t=%g" % T_SPLIT,
        run_with_late_dose(
            sim, t_eval, {"clonidine": CLONIDINE_nM}, {"yohimbine": YOHIMBINE_nM}
        ),
    )
    record(
        "clonidine, then 5x yohimbine at t=%g" % T_SPLIT,
        run_with_late_dose(
            sim,
            t_eval,
            {"clonidine": CLONIDINE_nM},
            {"yohimbine": 5.0 * YOHIMBINE_nM},
        ),
    )
    tone_na_clonidine, _ = record(
        "noradrenaline + clonidine",
        run(sim, t_eval, noradrenaline=NORADRENALINE_nM, clonidine=CLONIDINE_nM),
    )

    peak = cortex_na.max() - 0.5
    print(
        "\ncortical response decomposed (peak %.4f above rest):\n"
        "  alpha1 part %.4f  (removed by prazosin)\n"
        "  beta1  part %.4f  (removed by propranolol)\n"
        "  both blocked leaves %.4f, i.e. %.1f%% of the response removed"
        % (
            peak,
            peak - (cortex_no_alpha1.max() - 0.5),
            peak - (cortex_no_beta.max() - 0.5),
            cortex_neither.max() - 0.5,
            100.0 * (1.0 - (cortex_neither.max() - 0.5) / peak),
        )
    )
    print(
        "the beta1 part is the larger one because noradrenaline binds beta1\n"
        "8.5-fold more tightly than alpha1A, not because its gain is larger:\n"
        "  ALPHA1_MECHANISM_GAIN = %g, BETA_MECHANISM_GAIN = %g"
        % (noradrenaline.ALPHA1_MECHANISM_GAIN, noradrenaline.BETA_MECHANISM_GAIN)
    )
    print(
        "\npartial agonism, both ways round: clonidine alone takes the LC to\n"
        "%.4f, and against a noradrenaline background it RAISES it from\n"
        "%.4f to %.4f, because a partial agonist displacing a full one is a\n"
        "functional antagonist (clonidine efficacy %.2f [Jasper1998])."
        % (
            tone_clonidine.min(),
            conditions[1][2].min(),
            tone_na_clonidine.min(),
            noradrenaline.ALPHA2_EFFICACY_CLONIDINE,
        )
    )
    print(
        "\nnote what is NOT here: clonidine suppresses the LC and changes\n"
        "nothing about cortical noradrenaline, because no circuit rate has a\n"
        "path back into the PK layer. In a real brain that suppression is\n"
        "exactly how clonidine reduces forebrain noradrenaline release. P6.1."
    )

    plot(conditions)


def plot(conditions):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not available; numbers only)")
        return

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for label, t, tone_trace, cortex_trace in conditions:
        axes[0].plot(t, tone_trace, label=label)
        axes[1].plot(t, cortex_trace, label=label)
    axes[0].set_ylabel("noradrenergic_tone (LC_NA rate)")
    axes[1].set_ylabel("cortical_output (PFC_Pyramidal rate)")
    axes[1].set_xlabel("time (min)")
    axes[0].legend(fontsize=7)
    fig.suptitle("alpha1, alpha2 and beta adrenoceptors on one LC/cortex circuit (P5.3)")
    out = os.path.join(
        os.path.dirname(__file__),
        "..",
        "results",
        "noradrenaline_locus_coeruleus_cortex.png",
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print("wrote %s" % os.path.normpath(out))


if __name__ == "__main__":
    main()
