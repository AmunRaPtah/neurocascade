"""Produce P2.3's identifiability artifacts under `results/identifiability/`.

Run with the project venv from the repository root:

    PYTHONPATH=src python scripts/run_identifiability.py

Five figures and three tables, in the order the analysis is argued in
`docs/architecture.md`'s P2.3 section:

  1. `structural_degeneracy.png` - two numerically very different amygdala
     parameter sets predicting the same thing, for every drug gain.
  2. `fisher_spectrum.png` - the Fisher information matrix's eigenvalues:
     one nonzero, eleven at machine zero.
  3. `profile_likelihood_amygdala.png` - per-parameter profiles in regimes A
     (only this parameter free) and B (this parameter and the drug gain).
  4. `reduced_profile_likelihood.png` - regime C, in the {A,B,C,D,gain}
     space, i.e. with every circuit parameter free at once.
  5. `posterior_correlation.png` - the same result from an MCMC, for a
     reader who prefers a scatter plot to a rank argument.

The MCMC panel is the slow part (a few minutes); pass `--no-mcmc` to skip
it while iterating on the rest.
"""

import argparse
import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from neurocascade import scenarios  # noqa: E402
from neurocascade.calibration import identifiability as ident  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "identifiability")

#: Parameters given a joint diagnostic MCMC, one per failure mode: a
#: connectivity weight (degenerate with the drug gain through {A,B,C,D}) and
#: a time constant (invisible to a steady-state observable at all).
MCMC_PARAMETERS = ("weight_CeA__BLA", "tau_BLA")

VERDICT_COLOURS = {
    "structurally-unidentifiable": "#7f7f7f",
    "structurally-degenerate": "#d62728",
    "practically-unidentifiable": "#ff7f0e",
    "one-sided-identifiable": "#1f77b4",
    "identifiable": "#2ca02c",
}


def _plot_profile(ax, grid, profile, colour, label=None):
    """Draw one profile with an x-axis that can actually be read.

    These grids span several orders of magnitude, are sometimes entirely
    negative, and sometimes contain an exact zero (the drug gain's
    mechanistic null). A symlog axis handles all three and is illegible in
    all three, so instead: negative-only grids are plotted against their
    magnitude on a log axis, and an exact zero is plotted as a separate
    marker at the left edge, since "the value at exactly zero" is a claim
    about one point, not part of a curve.
    """
    grid = np.asarray(grid, dtype=float)
    profile = np.asarray(profile, dtype=float)
    zero = grid == 0.0
    nonzero = ~zero
    magnitude = np.abs(grid[nonzero])
    negative = np.all(grid[nonzero] < 0.0)
    positive = np.all(grid[nonzero] > 0.0)

    if negative or positive:
        ax.plot(magnitude, profile[nonzero], color=colour, lw=1.6, label=label)
        ax.set_xscale("log")
        if zero.any():
            ax.plot([magnitude.min()], [profile[zero][0]], marker="v", ms=7,
                    color=colour, ls="none")
            ax.annotate("= 0", (magnitude.min(), profile[zero][0]),
                        textcoords="offset points", xytext=(4, 4), fontsize=7,
                        color=colour)
        return "|value|" if negative else "value"

    ax.plot(grid, profile, color=colour, lw=1.6, label=label)
    return "value"


def _save(fig, name):
    path = os.path.join(RESULTS_DIR, name)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.relpath(path))


def figure_structural_degeneracy():
    """The eight non-tau parameters collapse into four combinations, so a
    completely different-looking circuit predicts exactly the same thing."""
    activation = ident.default_activation()
    equivalent = ident.equivalent_parameterisation()
    gains = np.linspace(0.0, 1.0, 41)
    nominal = np.array([ident.predicted_anxiety_ratio(None, g, activation) for g in gains])
    relabelled = np.array(
        [ident.predicted_anxiety_ratio(equivalent, g, activation) for g in gains])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(gains, nominal, lw=3, color="#1f77b4", label="scenarios.py parameters")
    axes[0].plot(gains, relabelled, lw=1.5, ls="--", color="#d62728",
                 label="a relabelled circuit with identical {A,B,C,D}")
    axes[0].axhline(ident.OBSERVED_RATIO, color="k", lw=1, ls=":")
    axes[0].fill_between(
        gains, ident.OBSERVED_RATIO - ident.OBSERVED_RATIO_SD,
        ident.OBSERVED_RATIO + ident.OBSERVED_RATIO_SD, color="k", alpha=0.10,
        label="the observation $\\pm$1 sd (SYNTHETIC)")
    axes[0].set_xlabel("gaba_a_connection_modulation_gain")
    axes[0].set_ylabel("anxiety-index drug/vehicle ratio")
    axes[0].set_title("Two circuits, one prediction")
    axes[0].legend(fontsize=8)

    axes[1].semilogy(gains, np.abs(nominal - relabelled) + 1e-18, color="#d62728")
    axes[1].axhline(1e-8, color="k", lw=1, ls=":")
    axes[1].set_xlabel("gaba_a_connection_modulation_gain")
    axes[1].set_ylabel("|difference| between the two curves")
    axes[1].set_title("Difference is integrator noise (dotted: rtol $10^{-8}$)")

    changed = {
        k: (ident.AMYGDALA_PARAMETERS_BY_NAME[k].nominal, v)
        for k, v in equivalent.items()
    }
    caption = "  ".join(
        "%s %.3g$\\rightarrow$%.3g" % (k.replace("_", " "), a, b)
        for k, (a, b) in list(changed.items())[:4]
    )
    fig.suptitle("Structural degeneracy: " + caption, fontsize=8, y=1.03)
    _save(fig, "structural_degeneracy.png")


def figure_fisher_spectrum():
    """One scalar observation gives a rank-one Fisher information matrix."""
    names = [p.name for p in ident.AMYGDALA_PARAMETERS]
    values, vectors = ident.fisher_spectrum(names)
    floor = 1e-20
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))

    axes[0].bar(range(len(values)), np.maximum(np.abs(values), floor),
                color=["#d62728"] + ["#7f7f7f"] * (len(values) - 1))
    axes[0].set_yscale("log")
    axes[0].set_xlabel("eigenvalue index")
    axes[0].set_ylabel("|eigenvalue| of the Fisher information matrix")
    axes[0].set_title("Rank 1: one datum, one degree of freedom")

    leading = vectors[:, 0]
    order = np.argsort(np.abs(leading))
    axes[1].barh(range(len(names)), leading[order], color="#1f77b4")
    axes[1].set_yticks(range(len(names)))
    axes[1].set_yticklabels([names[i].replace("_", " ") for i in order], fontsize=7)
    axes[1].set_xlabel("component of the one informed eigenvector")
    axes[1].set_title("The single direction the data sees")
    _save(fig, "fisher_spectrum.png")


def figure_parameter_profiles(report, grid_points):
    activation = ident.default_activation()
    params = ident.AMYGDALA_PARAMETERS
    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    by_name = {row["parameter"]: row for row in report}

    for ax, param in zip(axes.ravel(), params):
        grid_a, profile_a, _ = ident.profile_chi_square(
            param.name, free_drug_gain=False, activation=activation, points=grid_points)
        axis_kind = _plot_profile(ax, grid_a, profile_a, "#1f77b4",
                                  "A: only this parameter free")
        if param.kind != "drug_gain":
            grid_b, profile_b, _ = ident.profile_chi_square(
                param.name, free_drug_gain=True, activation=activation,
                points=grid_points)
            _plot_profile(ax, grid_b, profile_b, "#ff7f0e", "B: + drug gain free")
        ax.axhline(ident.PROFILE_THRESHOLD_CHI2, color="k", ls=":", lw=1)
        ax.axvline(abs(param.nominal) if axis_kind == "|value|" else param.nominal,
                   color="#7f7f7f", ls="--", lw=1)
        ax.set_ylim(-0.5, 25)
        verdict = by_name[param.name]["verdict"]
        ax.set_title("%s%s\n%s" % (param.name.replace("_", " "),
                                   " (magnitude)" if axis_kind == "|value|" else "",
                                   verdict),
                     fontsize=8, color=VERDICT_COLOURS[verdict])
        ax.tick_params(labelsize=7)
        ax.set_ylabel("$\\Delta\\chi^2$", fontsize=8)

    axes.ravel()[0].legend(fontsize=7)
    fig.suptitle(
        "Profile likelihood, one behavioural observation (SYNTHETIC stand-in). "
        "Dotted: 95% threshold $\\Delta\\chi^2$ = 3.84. Dashed: the "
        "scenarios.py value.\nRegime A's apparent bounds are the other ten "
        "illustrative numbers doing the work; freeing one more parameter "
        "(regime B) already removes most of them.",
        fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, "profile_likelihood_amygdala.png")


def figure_reduced_profiles(reduced_rows, grid_points):
    activation = ident.default_activation()
    fig, axes = plt.subplots(1, len(ident.REDUCED_PARAMETERS), figsize=(17, 3.6))
    by_name = {row["combination"]: row for row in reduced_rows}
    for ax, name in zip(axes, ident.REDUCED_PARAMETERS):
        grid, profile = ident.profile_reduced(
            name, points=grid_points, activation=activation)
        axis_kind = _plot_profile(ax, grid, profile, "#2ca02c")
        ax.axhline(ident.PROFILE_THRESHOLD_CHI2, color="k", ls=":", lw=1)
        verdict = by_name[name]["verdict"]
        label = name if len(name) < 12 else "drug gain"
        ax.set_title("%s%s\n%s" % (label,
                                   " (magnitude)" if axis_kind == "|value|" else "",
                                   verdict),
                     fontsize=9, color=VERDICT_COLOURS[verdict])
        ax.set_ylim(-0.5, 20)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("$\\Delta\\chi^2$")
    fig.suptitle(
        "Regime C: profiles in the reduced {A, B, C, D, gain} space, which is "
        "the same thing as profiling with EVERY circuit parameter free. "
        "Nothing is localised; the only\nunconditional statement the datum "
        "supports is that the drug gain is not zero (gain = 0 predicts a "
        "ratio of exactly 1 for every possible circuit, $\\Delta\\chi^2$ = "
        "%.1f)." % ident.chi_square(1.0),
        fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    _save(fig, "reduced_profile_likelihood.png")


def figure_posterior_correlation(draws, tune, chains):
    stats_rows = []
    fig, axes = plt.subplots(1, len(MCMC_PARAMETERS), figsize=(11, 4.4))
    for ax, name in zip(np.atleast_1d(axes), MCMC_PARAMETERS):
        print("sampling joint diagnostic model for", name)
        idata, summary, statistics = ident.run_joint_diagnostic(
            name, draws=draws, tune=tune, chains=chains)
        stats_rows.append(statistics)
        gain = np.asarray(
            idata.posterior["gaba_a_connection_modulation_gain"]).ravel()
        other = np.asarray(idata.posterior[name]).ravel()
        ax.scatter(other, gain, s=3, alpha=0.15, color="#1f77b4", edgecolors="none")
        lower, upper = ident.diagnostic_prior_bounds(name)
        ax.set_xlim(lower, upper)
        ax.set_xlabel(name.replace("_", " "))
        ax.set_ylabel("gaba_a_connection_modulation_gain")
        ax.set_title(
            "%s\nposterior corr %.2f, posterior sd / prior sd %.2f"
            % (name, statistics["posterior_correlation"],
               statistics["posterior_sd_over_prior_sd"]),
            fontsize=9)
    fig.suptitle(
        "The same non-identifiability seen as a posterior. Left: a "
        "connectivity weight trades off against the drug gain along a "
        "ridge.\nRight: a time constant's posterior is its prior, because "
        "the observable does not depend on it at all.", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    _save(fig, "posterior_correlation.png")
    return stats_rows


def write_tables(report, reduced_rows, mcmc_rows):
    path = os.path.join(RESULTS_DIR, "identifiability_verdicts.csv")
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "parameter", "kind", "nominal", "scan_low", "scan_high",
            "sensitivity", "interval_only_this_free",
            "interval_this_and_drug_gain_free", "interval_whole_circuit_free",
            "verdict", "scan_basis",
        ])
        for row in report:
            writer.writerow([
                row["parameter"], row["kind"], row["nominal"],
                row["scan_low"], row["scan_high"], row["sensitivity"],
                row["profile_interval_gain_fixed"],
                row["profile_interval_gain_free"],
                row["profile_interval_circuit_free"],
                row["verdict"], row["scan_basis"],
            ])
    print("wrote", os.path.relpath(path))

    path = os.path.join(RESULTS_DIR, "reduced_space_profiles.csv")
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["combination", "box_low", "box_high",
                         "profile_interval", "interval_span_ratio",
                         "bounds_at_searched_edge", "verdict", "meaning"])
        for row in reduced_rows:
            writer.writerow([row["combination"], row["box_low"], row["box_high"],
                             row["profile_interval"], row["interval_span_ratio"],
                             row["bounds_at_searched_edge"],
                             row["verdict"], row["meaning"]])
    print("wrote", os.path.relpath(path))

    if mcmc_rows:
        path = os.path.join(RESULTS_DIR, "joint_posterior_diagnostics.csv")
        with open(path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(mcmc_rows[0].keys()))
            writer.writeheader()
            for row in mcmc_rows:
                writer.writerow(row)
        print("wrote", os.path.relpath(path))


def write_summary(report, reduced_rows, mcmc_rows):
    lines = []
    lines.append("P2.3 identifiability of circuit-layer parameters")
    lines.append("=" * 62)
    lines.append("")
    lines.append("Behavioural observations available to the circuit layer: %d"
                 % sum(ident.behavioural_observation_counts().values()))
    lines.append("  %s" % ident.GABA_A_OBSERVATION_LABEL)
    lines.append("  target %.4f +/- %.4f (drug/vehicle anxiety ratio)"
                 % (ident.OBSERVED_RATIO, ident.OBSERVED_RATIO_SD))
    lines.append("")
    values, _ = ident.fisher_spectrum()
    lines.append("Fisher information eigenvalues (12 parameters, 1 observation):")
    lines.append("  " + "  ".join("%.3g" % v for v in values))
    lines.append("  -> rank 1: at most one direction in circuit-parameter "
                 "space is identifiable.")
    lines.append("")
    lines.append("Per-parameter verdicts (amygdala anxiety circuit):")
    width = max(len(row["parameter"]) for row in report)
    for row in report:
        lines.append("  %-*s  %-28s sensitivity %10.3e"
                     % (width, row["parameter"], row["verdict"], row["sensitivity"]))
    lines.append("")
    lines.append("Reduced-space profiles (every circuit parameter free at once).")
    lines.append("'edge' marks an endpoint that sits in the outermost searched "
                 "cell, i.e. a bound")
    lines.append("set by where the scan box stops rather than by the data:")
    for row in reduced_rows:
        edges = ["edge" if flag else "interior"
                 for flag in row["bounds_at_searched_edge"]]
        lines.append("  %-34s %-26s 95%% interval %s  [%s / %s]"
                     % (row["combination"], row["verdict"],
                        row["profile_interval"], edges[0], edges[1]))
    lines.append("")
    lines.append("Circuits with no behavioural observation at all "
                 "(Fisher information identically zero):")
    for row in ident.circuits_without_behavioural_evidence():
        lines.append("  %-38s %d circuit-layer parameters, all unidentifiable"
                     % (row["circuit"], row["circuit_parameters"]))
    if mcmc_rows:
        lines.append("")
        lines.append("Joint diagnostic posteriors:")
        for row in mcmc_rows:
            lines.append(
                "  %-20s corr(gain, param) = %+.3f  posterior sd / prior sd = "
                "%.3f  max r_hat %.3f  min ESS %.0f"
                % (row["parameter"], row["posterior_correlation"],
                   row["posterior_sd_over_prior_sd"], row["max_r_hat"],
                   row["min_ess_bulk"]))
    lines.append("")
    lines.append("Conclusion: no connectivity weight and no time constant is "
                 "identifiable from the")
    lines.append("behavioural evidence available. All of them keep their "
                 "illustrative label in")
    lines.append("scenarios.py. The one unconditional statement the datum "
                 "supports is that the")
    lines.append("drug-action gain is not zero (a gain of zero predicts a "
                 "ratio of exactly 1 for")
    lines.append("every possible circuit; delta chi-square = %.2f, i.e. %.1f "
                 "sd)." % (ident.chi_square(1.0), np.sqrt(ident.chi_square(1.0))))
    text = "\n".join(lines) + "\n"
    path = os.path.join(RESULTS_DIR, "summary.txt")
    with open(path, "w") as handle:
        handle.write(text)
    print("wrote", os.path.relpath(path))
    print()
    print(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-points", type=int, default=41)
    parser.add_argument("--no-mcmc", action="store_true")
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--tune", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=2)
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("scenarios.py amygdala gain (ILLUSTRATIVE since P11.3): %.4f"
          % scenarios.GABA_A_CONNECTION_MODULATION_GAIN)

    figure_structural_degeneracy()
    figure_fisher_spectrum()

    report = ident.identifiability_report(grid_points=args.grid_points)
    figure_parameter_profiles(report, args.grid_points)

    reduced_rows = ident.reduced_report(grid_points=args.grid_points)
    figure_reduced_profiles(reduced_rows, args.grid_points)

    mcmc_rows = []
    if not args.no_mcmc:
        mcmc_rows = figure_posterior_correlation(args.draws, args.tune, args.chains)

    write_tables(report, reduced_rows, mcmc_rows)
    write_summary(report, reduced_rows, mcmc_rows)


if __name__ == "__main__":
    main()
