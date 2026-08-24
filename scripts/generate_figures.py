"""P10.1: generate the reproducible NeuroCascade figure set.

Run from the repository root with the project environment, for example::

    /root/projects/pardalos/.venv/bin/python scripts/generate_figures.py

The outputs are written to ``results/figures``. The region activity figure is
an atlas-ordered CCF region strip: this repository has no section-image
coordinates, so it does not pretend that a heatmap is an anatomical brain
section. The remaining figures use the committed PK, endpoint and held-out
validation paths directly.
"""

import argparse
import csv
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from neurocascade import scenarios  # noqa: E402
from neurocascade.legacy import drug_pk
from neurocascade.calibration import endpoint_calibration as ec  # noqa: E402
from neurocascade.calibration import endpoint_forward as ef  # noqa: E402
from neurocascade.calibration import heldout_forward as hf  # noqa: E402
from neurocascade.legacy.connectome.ontology import MAJOR_DIVISION_IDS  # noqa: E402
from neurocascade.engine import Simulation  # noqa: E402
from neurocascade.receptor import Ligand, ReceptorSite  # noqa: E402


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIGURES = os.path.join(ROOT, "results", "figures")
HELDOUT_ERRORS = os.path.join(ROOT, "results", "heldout", "heldout_errors.csv")
FIGURE_NAMES = (
    "region_activity_map.png",
    "occupancy_time_course.png",
    "dose_response_curves.png",
    "heldout_validation_scatter.png",
)


def _save(fig, name):
    path = os.path.join(FIGURES, name)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return path


def _region_order(model):
    """Return node indices grouped in published major-division order."""
    division_order = {division_id: i for i, division_id in enumerate(MAJOR_DIVISION_IDS)}
    groups = {}
    for i, node in enumerate(model.network.nodes):
        structure_id = model.ontology.id_for_acronym(node.name)
        division_id = model.ontology.major_division(structure_id)
        groups.setdefault(division_order.get(division_id, len(division_order)), []).append(i)
    order = []
    labels = []
    boundaries = []
    for group_index in sorted(groups):
        members = sorted(
            groups[group_index],
            key=lambda i: model.ontology.id_for_acronym(model.network.nodes[i].name),
        )
        start = len(order)
        order.extend(members)
        division_id = MAJOR_DIVISION_IDS[group_index] if group_index < len(MAJOR_DIVISION_IDS) else None
        labels.append(
            model.ontology[division_id].acronym if division_id is not None else "other"
        )
        boundaries.append((start, len(order)))
    return order, labels, boundaries


def region_activity_map():
    """Generate baseline versus illustrative D2 activity by CCF region."""
    model, _ = hf.build_multi_target_model(
        ("D2",), "p10.1-region-activity", resolution="regions"
    )
    baseline = model.baseline_rates
    active = model.rates({"D2": 0.5})
    order, labels, boundaries = _region_order(model)
    matrix = np.vstack((baseline[order], active[order]))

    fig, ax = plt.subplots(figsize=(14, 2.9))
    image = ax.imshow(
        matrix, aspect="auto", interpolation="nearest", cmap="viridis",
        vmin=0.0, vmax=1.0,
    )
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["baseline", "D2 x = 0.5"])
    centers = [(start + stop - 1) / 2.0 for start, stop in boundaries]
    ax.set_xticks(centers)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    for _, stop in boundaries[:-1]:
        ax.axvline(stop - 0.5, color="white", lw=0.6, alpha=0.75)
    ax.set_xlabel("CCF regions ordered within major divisions")
    ax.set_title(
        "Region-wise whole-brain activity\n"
        "CCF parcellation strip; not an anatomical section image",
        fontsize=11,
    )
    colorbar = fig.colorbar(image, ax=ax, pad=0.02)
    colorbar.set_label("circuit rate")
    fig.text(
        0.01, -0.02,
        "Illustrative D2 perturbation; x is the mechanism product, not a fitted dose.",
        fontsize=8, color="0.35",
    )
    return _save(fig, "region_activity_map.png")


def occupancy_time_course():
    """Generate a dose-to-brain-concentration-to-occupancy time course."""
    pk = drug_pk.diazepam_rat_ip_pk(body_weight_kg=0.25)
    site = ReceptorSite(
        name="benzodiazepine_site",
        ligands=[Ligand(name="diazepam", kon=0.2, koff=0.2 * 3.6, efficacy=1.0)],
    )
    simulation = Simulation(
        pk_models=[pk],
        receptor_site=site,
        network=scenarios.amygdala_anxiety_circuit_network(),
        coupling={},
        connection_modulation={},
    )
    times = np.linspace(0.0, 240.0, 481)
    result = simulation.run(
        doses=[2.0], t_span=(0.0, 240.0), t_eval=times,
    )
    concentration = result["receptor_concentrations"][0]
    occupancy = result["occupancy"][0]

    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    ax.plot(times, occupancy, color="#2c6e9e", lw=2.2, label="benzodiazepine-site occupancy")
    ax.set_xlabel("time after 2 mg/kg IP dose (min)")
    ax.set_ylabel("occupancy fraction")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(axis="y", color="0.88", lw=0.8)
    ax2 = ax.twinx()
    ax2.plot(times, concentration, color="#c46b35", lw=1.7, ls="--", label="unbound brain concentration")
    ax2.set_ylabel("unbound brain diazepam (nM)", color="#c46b35")
    ax2.tick_params(axis="y", colors="#c46b35")
    handles, labels = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles + handles2, labels + labels2, loc="center right", frameon=False)
    ax.set_title("Diazepam brain exposure and receptor occupancy")
    fig.text(
        0.01, -0.01,
        "Rat IP PK/partition path; single-ligand illustrative receptor trace.",
        fontsize=8, color="0.35",
    )
    return _save(fig, "occupancy_time_course.png")


def dose_response_curves():
    """Plot synthetic graded observations against the model's upper bound.

    P10.1 plotted the two graded rodent datasets ([Sagara2008] locomotion,
    [Lustyk2023] analgesia) here; P11.3 retired them together with their
    dataset (`legacy/behavioral_evidence.py`). The points below are
    therefore generated FROM the model itself at a known gain, through
    `ec.synthetic_target_from_curve`, so the figure demonstrates that the
    fit pipeline still runs end to end rather than comparing anything with
    the world.
    """
    endpoint_keys = ("locomotion", "analgesia")
    model, _ = ef.build_whole_brain_model(
        "Gabra1", "connection_modulation", "p10.1-dose-response",
        resolution="regions",
    )
    curve = model.response_curve(list(endpoint_keys), np.linspace(0.0, 5.0, 31))
    exposures_nm = (10.0, 30.0, 100.0)
    truth_gains = {"locomotion": 1.5, "analgesia": 2.0}
    targets = [
        ec.synthetic_target_from_curve(
            curve, key, gain=truth_gains[key],
            exposures_nM=exposures_nm, sds=(0.02,) * len(exposures_nm),
        )
        for key in endpoint_keys
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    for ax, target in zip(axes, targets):
        brain_nm = np.asarray([level[1] for level in target.levels])
        observed, observed_sd = target.endpoint_ratio()
        predicted = ec.predicted_ratio(curve, target, gain=5.0)
        ax.errorbar(
            brain_nm, observed, yerr=observed_sd, fmt="o", ms=5, capsize=3,
            color="#b03030", label="synthetic ratio (gain %.1f)" % truth_gains[target.endpoint_key],
        )
        ax.plot(
            brain_nm, predicted, color="#2c6e9e", lw=2, marker="s",
            label="model, gain = 5 upper bound",
        )
        ax.axhline(1.0, color="0.45", lw=1, ls="--")
        ax.set_xscale("log")
        ax.set_xlabel("unbound brain diazepam (nM, synthetic)")
        ax.set_ylabel("endpoint index ratio")
        ax.set_title(target.endpoint_key.capitalize())
        ax.grid(axis="y", color="0.88", lw=0.8)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle(
        "Synthetic graded observations versus the whole-brain model "
        "(no published graded target remains)",
        fontsize=12,
    )
    fig.text(
        0.01, -0.02,
        "Points are synthetic, generated from the model itself at known gains; "
        "the model line uses the prior upper bound, not a validated whole-brain gain.",
        fontsize=8, color="0.35",
    )
    return _save(fig, "dose_response_curves.png")


def _single_float(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def heldout_validation_scatter():
    """Plot quantitative held-out ratios against reachable model intervals."""
    rows = []
    with open(HELDOUT_ERRORS, newline="") as handle:
        for row in csv.DictReader(handle):
            observed = _single_float(row["observed_ratio"])
            low = _single_float(row["reachable_lo"])
            high = _single_float(row["reachable_hi"])
            if observed is None or low is None or high is None:
                continue
            rows.append((row, observed, low, high))
    if not rows:
        raise RuntimeError("held-out error table has no single-value quantitative rows")

    x = np.array([(low + high) / 2.0 for _, _, low, high in rows])
    xerr = np.array([(high - low) / 2.0 for _, _, low, high in rows])
    y = np.array([observed for _, observed, _, _ in rows])
    matched = np.array([row["direction_match"] == "yes" for row, *_ in rows])

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    for mask, color, label in (
        (matched, "#3a8f5b", "direction matched"),
        (~matched, "#b03030", "direction did not match"),
    ):
        ax.errorbar(
            x[mask], y[mask], xerr=xerr[mask], fmt="o", ms=5, capsize=2,
            color=color, ecolor=color, label=label,
        )
    lims = [min(0.45, y.min() - 0.05), max(1.8, y.max() + 0.05)]
    ax.plot(lims, lims, color="0.5", ls="--", lw=1, label="identity")
    for (row, observed, low, high), xpos in zip(rows, x):
        ax.annotate(
            "%s / %s" % (row["compound"], row["endpoint"]),
            (xpos, observed), xytext=(4, 4), textcoords="offset points",
            fontsize=7, color="0.25",
        )
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("midpoint of model reachable interval (horizontal bar = interval)")
    ax.set_ylabel("published observed endpoint ratio")
    ax.set_title("Held-out quantitative validation: reachable-range diagnostic")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.grid(color="0.9", lw=0.7)
    fig.text(
        0.01, -0.01,
        "This is not a fitted prediction: held-out gains remain unidentified and several chains are incomplete.",
        fontsize=8, color="0.35",
    )
    return _save(fig, "heldout_validation_scatter.png")


def generate():
    os.makedirs(FIGURES, exist_ok=True)
    paths = [
        region_activity_map(),
        occupancy_time_course(),
        dose_response_curves(),
        heldout_validation_scatter(),
    ]
    metadata = {
        "figures": [os.path.relpath(path, ROOT) for path in paths],
        "notes": [
            "Region activity is a CCF-ordered region strip because no atlas section coordinates are stored.",
            "Dose-response points are synthetic targets generated from the model at known gains, since P11.3 retired the rodent datasets; the figure is a pipeline demonstration and not a fitted prediction.",
            "Held-out scatter is a reachable-range diagnostic, not a fitted prediction.",
        ],
    }
    with open(os.path.join(FIGURES, "README.json"), "w") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    paths = generate()
    print("wrote %d figures to %s" % (len(paths), FIGURES))


if __name__ == "__main__":
    main()
