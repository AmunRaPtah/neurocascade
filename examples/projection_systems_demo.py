"""Neuromodulatory projection systems (P6.1).

Runs the chain the P6.1 brief names, end to end and in one coupled ODE
system:

    source nucleus firing rate
      -> transmitter release, weighted by projection density
      -> region-wise extracellular transmitter concentration
      -> receptor occupancy AT EVERY TARGET REGION
      -> drive at every one of those regions' nodes
      -> circuit rates -> readout

Four parts.

  Part 1, the volume-transmission timescales. The abstraction says a region
  has ONE transmitter concentration. That is a spatial approximation and this
  part is the argument for it, computed: the mean spacing between dopamine
  release sites is ~2 um ([Dreyer2010], cross-checked by [Ozcete2024]),
  diffusion crosses that in ~2 ms, and every transmitter's extracellular
  clearance time constant is at least 40-fold below the circuit's. Diffusion
  does NOT mix a whole region (518 s across a millimetre) and this model does
  not claim it does.

  Part 2, the field. The serotonin projection with the dorsal and median
  raphe as sources and the cortex as a target, run at three held firing
  rates, showing the concentration following the nucleus.

  Part 3, THE LOOP CLOSED. 8-OH-DPAT at the somatodendritic 5-HT1A
  autoreceptor lowers raphe firing, which lowers release, which lowers
  CORTICAL serotonin, which lowers cortical 5-HT1A occupancy, which lowers
  cortical firing. Every module from P5.1 to P5.8 had to record that second
  half as impossible: `serotonin.py` and `noradrenaline.py` each carry a test
  asserting that their own hand-built scenario cannot express it, and both
  still pass, because a port is a new object rather than an edit.

  Part 4, tonic transmitter at every target at once. An antagonist given
  alone was inert in every module from P5.1 to P5.7 and stopped being inert
  for one transmitter at one site in P5.8. Here it is not inert for three
  transmitters across a projection field, and the two where it still looks
  inert are the two whose basal concentration is a UNIT rather than a
  measurement, which is a gap in the literature rather than in the model.

Species: mouse for the dopamine and serotonin uptake kinetics and basal
levels, rat for the noradrenaline clearance rate and the histamine basal
level, and `projection_systems.py` states every substitution.

Run with the shared Pardalos venv:
    /root/projects/pardalos/.venv/bin/python examples/projection_systems_demo.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from neurocascade.legacy import projection_scenarios as pscen  # noqa: E402
from neurocascade.legacy import projection_systems as ps  # noqa: E402


DOSE_NM = 100.0


def rule(title):
    print()
    print(title)
    print("=" * 72)


def part_1_timescales():
    rule("Part 1: the volume-transmission timescales")
    print(ps.timescale_report())


def part_2_the_field():
    rule("Part 2: the field follows the nucleus")
    scenario = pscen.port("serotonin")
    field = scenario.projection
    print(
        "Sources: %s. Target regions: %s."
        % (", ".join(field.source_nodes), ", ".join(field.regions))
    )
    print(
        "Density map is %s, so every region sits at the reference region's"
        % pscen.DENSITY_PROVENANCE
    )
    print("measured basal level of %.2f nM ([Mathews2004])." % field.basal_concentration("PFC"))
    print()
    print("%-24s %14s %12s" % ("raphe firing (rest=0.5)", "cortical 5-HT", "fold"))
    basal = field.basal_concentration("PFC")
    for rate in (0.25, 0.5, 1.0):
        rates = [rate] * len(field.sources)
        value = field.steady_state("PFC", rates)
        print("%-24.2f %11.3f nM %11.2fx" % (rate, value, value / basal))
    print()
    print("Near-proportional, and convex: doubling the firing rate takes the")
    print("field to 2.03x rather than 2.00x, because Michaelis-Menten uptake")
    print("has less headroom the harder it is pushed. The departure is 1-2%")
    print("because 2.8 nM sits far below the measured Km of 170 nM")
    print("([JohnJones2007]).")


def part_3_the_loop():
    rule("Part 3: the autoreceptor loop, closed")
    scenario = pscen.port("serotonin")
    net0, sim0, base = scenario.run()
    net1, sim1, dosed = scenario.run(**{"8-OH-DPAT": DOSE_NM})

    def at_end(network, sim, result):
        return {
            "DRN_5HT": scenario.rate_trace(network, result, "DRN_5HT")[-1],
            "cortical 5-HT (nM)": scenario.field_trace(sim, result, "PFC")[-1],
            "cortical 5-HT1A occ": scenario.occupancy_trace(
                sim, result, "5HT1A", "PFC", "serotonin"
            )[-1],
            "PFC_Pyramidal": scenario.rate_trace(network, result, "PFC_Pyramidal")[-1],
        }

    before = at_end(net0, sim0, base)
    after = at_end(net1, sim1, dosed)
    print("%-22s %12s %12s %12s" % ("", "no drug", "8-OH-DPAT", "change"))
    for key in before:
        print(
            "%-22s %12.4f %12.4f %+12.4f"
            % (key, before[key], after[key], after[key] - before[key])
        )
    print()
    print("Links 2 to 4 are the ones that did not exist before P6.1: the")
    print("cortical serotonin concentration is produced by a release term that")
    print("reads DRN_5HT's rate out of the shared state vector.")

    print()
    print("The same probe on every port that has one:")
    print(
        "%-14s %-12s %10s %12s %12s %11s"
        % ("port", "drug", "source r", "field fold", "occupancy", "terminal r")
    )
    for key in pscen.PORT_ORDER:
        if key not in pscen.AUTORECEPTOR_PROBE:
            continue
        effect = pscen.closed_loop_effect(key)
        print(
            "%-14s %-12s %+10.4f %+12.4f %+12.2e %+11.4f"
            % (
                key,
                effect["drug"],
                effect["source_rate"],
                effect["terminal_field_fold"],
                effect["terminal_occupancy"],
                effect["terminal_rate"],
            )
        )
    print()
    print("Noradrenaline's and acetylcholine's fields move and their receptors")
    print("do not: both carry a UNIT basal scale, because no rodent basal")
    print("extracellular concentration was found for either (see")
    print("projection_systems.py, 'Searched and empty'). 1.0 nM is three to")
    print("four orders below any Kd in this codebase.")


def part_4_tonic_transmitter():
    rule("Part 4: an antagonist given alone is no longer inert")
    print("%-14s %-14s %12s  %s" % ("port", "antagonist", "terminal r", "basal scale"))
    for key in pscen.PORT_ORDER:
        effect = pscen.lone_antagonist_effect(key)
        print(
            "%-14s %-14s %+12.5f  %s"
            % (
                key,
                effect["drug"],
                effect["terminal_rate"],
                "measured" if key in pscen.MEASURED_BASAL else "UNIT",
            )
        )
    print()
    print("Tonic occupancy at rest, and the standing drive it puts on a node:")
    for key in pscen.PORT_ORDER:
        scenario = pscen.port(key)
        if not scenario.standing_drive:
            continue
        terms = ", ".join(
            "%s %+.4f" % (node, value)
            for node, value in sorted(scenario.standing_drive.items())
        )
        print("  %-14s %s" % (key, terms))
    print()
    print("`projection.compensate_baselines` folds those into the baselines,")
    print("which is P3.4's lesson for the fourth time: any mechanism that adds")
    print("a standing term to a node's drive invalidates a stability")
    print("certificate computed without it.")


def main():
    part_1_timescales()
    part_2_the_field()
    part_3_the_loop()
    part_4_tonic_transmitter()
    print()
    print("Full report: scripts/report_projection_systems.py ->")
    print("results/projection_systems.txt")


if __name__ == "__main__":
    main()
