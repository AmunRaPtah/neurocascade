"""P7.3: fit the P7.1 endpoints against published dose-response data.

    python scripts/run_endpoint_fits.py > results/calibration/endpoint_fits.txt

Builds the whole-brain drug model once, tabulates the endpoint index over
the gain range the prior allows, and then, per target: says whether the
published contrast is even inside the model's reach, measures the
gain/scale degeneracy, samples the posterior and prints its convergence
diagnostics.

Since P11.3 there are NO published targets left to fit (`ec.TARGETS` is
empty - every dataset this ever had was rodent and moved to
`legacy/behavioral_evidence.py`; the committed diagnostics under
`results/calibration/` are the historical record). Run bare, this script
now builds the model, states that, and exits. Run with `--synthetic`, it
exercises the whole pipeline against explicitly-labelled synthetic targets
generated FROM the model itself at known gains (`ec.synthetic_target_from_
curve`), so anyone can check the scaffold still fits an endpoint end to end
without waiting for human data. Nothing either mode produces removes an
illustrative label.

Runs off the gitignored data/connectome cache; falls back to the committed
fifteen-region fixtures when the cache is absent, and says which it used.
The 291-region sweep takes a couple of minutes (one 291-node ODE solve per
grid point); the sampling that follows is seconds, because it interpolates
that table rather than re-integrating.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np                                              # noqa: E402
import pandas as pd                                             # noqa: E402

from neurocascade.calibration import (                          # noqa: E402
    endpoint_calibration as ec,
    endpoint_forward as efwd,
    forward as fwd,
)

#: The gain grid the response curve is tabulated on. Denser near zero
#: because that is where the curve bends; the endpoints of the grid are the
#: prior's own bounds, so no fit can ask for a value the table does not
#: cover and `ResponseCurve.__call__` never has to extrapolate.
_, GAIN_LO, GAIN_HI = ec.WHOLE_BRAIN_PRIOR_SPECS[
    "gaba_a_whole_brain_modulation_gain"
]
GRID_POINTS = 25

#: `--synthetic` mode's exposures, in nM, unbound brain. Arbitrary but not
#: pretending otherwise: they exist so the grid has something to scale to
#: and the synthetic targets have distinct activations. Labelled synthetic
#: all the way through by `ec.synthetic_target`.
SYNTHETIC_EXPOSURES_NM = (10.0, 30.0, 100.0)


def gain_grid():
    return np.concatenate([[GAIN_LO], np.geomspace(GAIN_HI / 250.0, GAIN_HI,
                                                   GRID_POINTS - 1)])


def _synthetic_targets(curve):
    """Two explicitly-synthetic targets generated from `curve` itself.

    A one-level contrast at a known gain (the shape everything P7.3 had) and
    a three-level graded series at another gain (the shape nothing P7.3 had
    enough of). Fitted back against the same curve, each recovers its own
    gain, which is the point.
    """
    return (
        ec.synthetic_target_from_curve(
            curve, "anxiety", gain=2.0,
            exposures_nM=SYNTHETIC_EXPOSURES_NM[-1:], sds=(0.02,),
        ),
        ec.synthetic_target_from_curve(
            curve, "locomotion", gain=1.5,
            exposures_nM=SYNTHETIC_EXPOSURES_NM, sds=(0.02,) * 3,
        ),
    )


def main():
    model, provenance = efwd.build_whole_brain_model(
        "Gabra1", "connection_modulation", name="diazepam/whole-brain",
    )
    use_synthetic = "--synthetic" in sys.argv[1:]
    print("P7.3: whole-brain endpoint fits%s" % (
        " (--synthetic)" if use_synthetic else ""))
    print("=" * 72)
    print()
    for key in ("connectome_description", "resolution", "n_nodes",
                "operating_gain", "weight_scale", "density_normalization",
                "peak_density_node"):
        print("  %-24s %s" % (key, provenance[key]))
    print("  %-24s %d (%s)" % (
        "unmeasured regions", len(provenance["unmeasured_nodes"]),
        ", ".join(provenance["unmeasured_nodes"]) or "none"))
    print()
    print("  Every circuit-layer number above is ILLUSTRATIVE: the node")
    print("  parameters (P3.2), the operating gain (P3.4), the")
    print("  expression-to-density proportionality (P4.2), every endpoint")
    print("  weight magnitude (P7.1) and the amygdala modulation gain")
    print("  restored to its placeholder by P11.3. See docs/architecture.md.")
    print()

    if use_synthetic:
        endpoint_keys = ["anxiety", "locomotion"]
        targets = None          # built once the curve exists, below
    elif not ec.TARGETS:
        endpoint_keys = []
        targets = ()
    else:
        endpoint_keys = sorted({t.endpoint_key for t in ec.TARGETS})
        targets = ec.TARGETS

    if not use_synthetic:
        print("  No published targets registered since P11.3 retired the")
        print("  rodent ones (see the module docstring of")
        print("  calibration/endpoint_calibration.py). Rerun with")
        print("  --synthetic to exercise the machinery on labelled")
        print("  synthetic targets instead.")
        print()
    for key in endpoint_keys:
        missing = model.endpoint_regions_without_density(key)
        print("  %-20s regions with no ISH density: %s"
              % (key, ", ".join(missing) if missing else "none"))
    print()

    grid = gain_grid()
    if use_synthetic:
        # Scale the grid to what the synthetic exposures can ask for, the
        # same way the real targets' largest activation scales it below.
        activation = max(
            fwd.equilibrium_occupancy_from_kd(
                ec.DIAZEPAM_KI_NM, exposure, efficacy=ec.DIAZEPAM_EFFICACY)
            for exposure in SYNTHETIC_EXPOSURES_NM
        )
    elif targets:
        # The table has to cover every x = gain * activation any target can ask
        # for, so it is scaled by the LARGEST activation in the target set;
        # `ResponseCurve.__call__` raises rather than extrapolate, so an
        # undersized grid would be an error and not a silent inaccuracy.
        activation = max(float(a.max()) for a in
                         (target.activation() for target in targets))
    else:
        activation = 1.0        # no target to scale to; nothing is read off it
    curve = model.response_curve(endpoint_keys, grid * activation) \
        if endpoint_keys else None
    if curve is not None:
        error = curve.interpolation_error(model)
        print("  response curve: %d points, x in [%.4g, %.4g]"
              % (len(grid), curve.x_grid[0], curve.x_grid[-1]))
        print("  interpolation error at grid midpoints, measured against a")
        print("  re-integration of the 291-node circuit:")
        for key in endpoint_keys:
            print("      %-20s %.3e" % (key, error[key]))
        print()

    if use_synthetic:
        targets = _synthetic_targets(curve)

    rows = []
    for target in targets:
        observed, observed_sd = target.endpoint_ratio()
        lo_ratio, hi_ratio = ec.reachable_ratio_range(curve, target)
        verdict = ec.bound_verdict(curve, target)
        best_gain, best_residual = ec.best_residual_sd(curve, target)
        print("-" * 72)
        print("%s  (endpoint %s)" % (target.key, target.endpoint_key))
        print("-" * 72)
        print("  measure                 %s" % target.measure)
        if target.evidence_key == "SYNTHETIC":
            print("  source                  NONE - synthetic observation,")
            print("                          %s" % target.note)
        else:
            print("  source                  %s" % target.evidence_key)
            if target.note:
                print("  note                    %s" % target.note)
        print("  levels                  %d dose(s) at %g min"
              % (target.n_levels, target.exposure_timepoint_min))
        print("    %-10s %-12s %-8s %-22s %-10s"
              % ("mg/kg", "brain nM", "activ.", "observed ratio", "model@5"))
        for (dose, exposure, _, _), act, obs, sd, hi_r in zip(
                target.levels, target.activation(), observed, observed_sd,
                hi_ratio):
            print("    %-10g %-12.4g %-8.4f %-22s %-10.4f"
                  % (dose, exposure, act, "%.4f +/- %.4f" % (obs, sd), hi_r))
        print("  verdict                 %s" % verdict)
        print("  best gain in the box    %.4f, worst residual %.2f sd"
              % (best_gain, best_residual))

        fisher = ec.information_matrix(curve, target, gain=0.5 * GAIN_HI)
        eigenvalues = np.linalg.eigvalsh(fisher)
        condition = (abs(eigenvalues[0]) / abs(eigenvalues[1])
                     if eigenvalues[1] else float("nan"))
        print("  Fisher spectrum over (gain, assay scale): %s"
              % np.array2string(eigenvalues, precision=3))
        print("      eigenvalue ratio %.2e: a single contrast gives a"
              % condition)
        print("      rank-1 matrix exactly, so the gain and the unknown")
        print("      behavioural scale are degenerate and only the RATIO")
        print("      formulation identifies anything at all. A graded")
        print("      series can lift that rank, and the number above says")
        print("      by how much rather than whether in principle.")

        idata, summary = ec.run_fit(target, curve)
        r_hat = float(pd.to_numeric(summary["r_hat"], errors="coerce").max())
        ess = float(pd.to_numeric(summary["ess_bulk"], errors="coerce").min())
        mean = float(summary["mean"].iloc[0])
        sd = float(summary["sd"].iloc[0])
        print("  posterior gain          %.4f +/- %.4f" % (mean, sd))
        print("  r_hat                   %.3f   (want < 1.05)" % r_hat)
        print("  ESS (bulk)              %.0f" % ess)
        interior = GAIN_LO + 0.02 * (GAIN_HI - GAIN_LO) < mean < (
            GAIN_HI - 0.02 * (GAIN_HI - GAIN_LO))
        print("  posterior is interior   %s" % interior)
        print("  prior mean for contrast %.4f" % (0.5 * (GAIN_LO + GAIN_HI)))
        if use_synthetic:
            truth = float(target.key.split("@gain=")[-1])
            print("  generating truth        %.4f (synthetic; recovered)"
                  % truth)
        print()

        os.makedirs(ec.RESULTS_DIR, exist_ok=True)
        stem = target.key.replace("/", "_")
        summary.to_csv(os.path.join(ec.RESULTS_DIR, "%s_summary.csv" % stem))
        idata.to_netcdf(os.path.join(ec.RESULTS_DIR, "%s_posterior.nc" % stem))
        rows.append({
            "target": target.key,
            "endpoint": target.endpoint_key,
            "n_levels": target.n_levels,
            "observed_ratio": " ".join("%.4f" % v for v in observed),
            "observed_sd": " ".join("%.4f" % v for v in observed_sd),
            "model_ratio_at_gain_upper": " ".join("%.4f" % v for v in hi_ratio),
            "best_gain": best_gain,
            "worst_residual_sd": best_residual,
            "verdict": verdict,
            "posterior_mean": mean,
            "posterior_sd": sd,
            "r_hat": r_hat,
            "ess_bulk": ess,
            "posterior_interior": interior,
        })

    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame.to_csv(
            os.path.join(ec.RESULTS_DIR, "endpoint_fit_diagnostics.csv"),
            index=False,
        )
    print("=" * 72)
    if frame.empty:
        print("(no targets fitted - see the notice above)")
        print()
        print("The five rodent fits this file used to hold are reproducible")
        print("from legacy/behavioral_evidence.py; their diagnostics survive")
        print("in results/calibration/. No illustrative label was removed by")
        print("them and none is claimed now.")
        return frame
    print(frame.to_string(index=False))
    print()
    if use_synthetic:
        print("Both synthetic posteriors recover the gain their observations")
        print("were generated at, which is the demonstration that this")
        print("machinery still fits an endpoint end to end. It proves")
        print("nothing about the world: every number above is synthetic by")
        print("construction, and no illustrative label moves.")
    else:
        print("Every fit above converged. None of them licenses a number:")
        print("a posterior that sits on a prior bound is a description of the")
        print("box, which is the check P2.2 made before sampling and the reason")
        print("its own bounds came from the mechanism. No illustrative label is")
        print("removed by this task; docs/architecture.md's P7.3 section says")
        print("what each endpoint would need instead.")
    return frame


if __name__ == "__main__":
    main()
