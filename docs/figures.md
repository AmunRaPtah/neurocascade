# P10.1: reproducible figures

Run [`scripts/generate_figures.py`](../scripts/generate_figures.py) from the
repository root with the project scientific environment. It writes four PNGs
and `README.json` to [`results/figures/`](../results/figures/):

- [`region_activity_map.png`](../results/figures/region_activity_map.png):
  baseline and illustrative D2-driven activity across the 291 CCF regions,
  ordered by major division. The repository contains no atlas section-image
  coordinates, so this is explicitly a parcellation strip rather than a
  fabricated anatomical section.
- [`occupancy_time_course.png`](../results/figures/occupancy_time_course.png):
  the rat diazepam IP PK/brain-partition path and single-ligand
  benzodiazepine-site occupancy.
- [`dose_response_curves.png`](../results/figures/dose_response_curves.png):
  the Sagara 2008 locomotion and Lustyk 2023 analgesia graded data against
  the whole-brain model at its gain upper bound. The line is a reachability
  diagnostic, not a validated fitted gain.
- [`heldout_validation_scatter.png`](../results/figures/heldout_validation_scatter.png):
  quantitative held-out observations against reachable-range midpoints, with
  horizontal bars showing the model interval. This is diagnostic because the
  held-out gains and several upstream chain links remain unavailable.

The script uses the model builders and committed CSV tables directly; no
figure data are copied into the plotting code. Re-running it overwrites the
same named outputs, making the figure set reproducible from the committed
script and current data cache.
