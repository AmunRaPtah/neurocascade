#!/usr/bin/env python
"""Regenerate the committed F-TRACT fixtures from the extracted cache.

Reads the extracted F-TRACT CSVs from data/human/ftract/extracted/ and
writes verbatim slices into src/neurocascade/human/fixtures/.

The six parcels selected for the slice were chosen to span both
hemispheres and guarantee: NaN cells (unprobed), asymmetric reciprocal
pairs, and the 0.0-vs-NaN distinction.

Also reads the TVB connectivity archives from /tmp/tvb/ (re-fetched if
absent) and writes tvb_signatures_fixture.json.

Usage::

    python scripts/make_human_ftract_fixture.py

    python scripts/make_human_ftract_fixture.py --tvb-dir /tmp/tvb
"""

import argparse
import json
import os
import sys
import zipfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402

FIXTURE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "src", "neurocascade", "human", "fixtures",
)

# The six parcels in the verbatim slice.  They span L/R hemispheres, both
# visual cortex (V1-V6) and guarantee an asymmetric reciprocal pair
# (L_V1->R_V1 != R_V1->L_V1 in probability) plus NaN cells.
SLICE_PARCELS = ["L_V1", "L_V6", "L_V2", "L_V3", "R_V1", "R_V6"]


def _read_raw_csv_lines(path):
    """Return (comment_lines, header_line, data_lines) as raw strings."""
    with open(path) as fh:
        lines = fh.read().splitlines()
    comment_lines = [l for l in lines if l.startswith("#")]
    header_line = lines[4]  # "stimulated parcels,..."
    data_lines = lines[5:]
    return comment_lines, header_line, data_lines


def _extract_slice(comment_lines, header_line, data_lines, parcels):
    """Return lines of a verbatim slice CSV for the selected parcels."""
    # Column labels from the header.
    col_labels = header_line.split(",")[1:]
    col_indices = [col_labels.index(p) for p in parcels]

    # Filter data rows to selected parcels.
    filtered = []
    for line in data_lines:
        row_label = line.split(",")[0]
        if row_label in parcels:
            parts = line.split(",")
            # Rebuild with "stimulated parcels" prefix + sliced columns.
            sliced = [row_label] + [parts[i + 1] for i in col_indices]
            filtered.append(",".join(sliced))

    # Build new header.
    new_header = "stimulated parcels," + ",".join(parcels)
    return comment_lines + [new_header] + filtered


def make_csv_fixture(name, cache_dir, parcels, fixture_dir):
    """Generate one CSV fixture file."""
    feature_dir = os.path.join(
        cache_dir, "ftract", "extracted", name
    )
    csv_path = os.path.join(feature_dir, "%s.csv" % name)
    if not os.path.exists(csv_path):
        print("SKIP %s: %s not found" % (name, csv_path))
        return

    comment_lines, header_line, data_lines = _read_raw_csv_lines(csv_path)
    slice_lines = _extract_slice(comment_lines, header_line, data_lines, parcels)
    out_path = os.path.join(fixture_dir, "ftract_%s_slice_fixture.csv" % name)
    with open(out_path, "w") as fh:
        fh.write("\n".join(slice_lines) + "\n")
    print("WROTE %s (%d lines)" % (out_path, len(slice_lines)))
    return out_path


def make_synaptic_fixture(feature_key, cache_dir, parcels, fixture_dir):
    """Generate one synaptic delay fixture (verbatim parcel subset)."""
    feature_dir = os.path.join(cache_dir, "ftract", "extracted", feature_key)
    txt_path = os.path.join(
        feature_dir,
        feature_key + ".txt",
    )
    if not os.path.exists(txt_path):
        print("SKIP %s: %s not found" % (feature_key, txt_path))
        return

    out_lines = []
    with open(txt_path) as fh:
        for line in fh:
            label = line.split("\t")[0].split()[0]
            if label in parcels:
                out_lines.append(line.rstrip("\n"))
    out_path = os.path.join(
        fixture_dir, "ftract_%s_slice_fixture.txt" % feature_key
    )
    with open(out_path, "w") as fh:
        fh.write("\n".join(out_lines) + "\n")
    print("WROTE %s (%d lines)" % (out_path, len(out_lines)))
    return out_path


def make_tvb_signatures(tvb_dir, fixture_dir):
    """Read the three TVB zip files and write tvb_signatures_fixture.json."""
    signatures = {}
    for n in (76, 96, 192):
        zip_name = "connectivity_%d.zip" % n
        zip_path = os.path.join(tvb_dir, zip_name)
        if not os.path.exists(zip_path):
            print("SKIP tvb_%d: %s not found" % (n, zip_path))
            continue

        with zipfile.ZipFile(zip_path) as zf:
            names = [n for n in zf.namelist() if not n.startswith("__MACOSX/") and not n.endswith(".DS_Store")]

        # Determine nested prefix.
        prefix = ""
        for name in names:
            if name.endswith("/weights.txt"):
                prefix = name.replace("weights.txt", "")
                break

        # Extract weights and labels.
        with zipfile.ZipFile(zip_path) as zf:
            with zf.open(prefix + "weights.txt") as f:
                weights = np.loadtxt(f)
            with zf.open(prefix + "centres.txt") as f:
                labels = [line.decode().strip().split()[0] for line in f if line.strip()]
            with zf.open(prefix + "info.txt") as f:
                info_text = f.read().decode("utf-8")

        uniq = np.unique(weights)
        label_prefixes = sorted({l.split("-")[0] for l in labels if "-" in l})

        sigs = {
            "shape": list(weights.shape),
            "n_unique": int(len(uniq)),
            "unique_values": [float(v) for v in sorted(uniq.tolist())] if len(uniq) <= 10 else None,
            "ordinal": bool(set(uniq.tolist()) <= {0.0, 1.0, 2.0, 3.0}),
            "symmetric": bool(np.array_equal(weights, weights.T)),
            "labels_glasser": all(
                l.startswith(("L_", "R_")) and len(l) > 2 for l in labels
            ) if labels else False,
            "label_prefixes": label_prefixes,
            "weights_unit": "au",
        }
        signatures[str(n)] = sigs
        print("TVB %d: shape=%s unique=%d ordinal=%s labels_glasser=%s prefixes=%s"
              % (n, sigs["shape"], sigs["n_unique"], sigs["ordinal"],
                 sigs["labels_glasser"], label_prefixes[:5]))

    out_path = os.path.join(fixture_dir, "tvb_signatures_fixture.json")
    with open(out_path, "w") as fh:
        json.dump(signatures, fh, indent=2)
        fh.write("\n")
    print("WROTE %s" % out_path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "human"),
        help="Path to data/human (default: ../data/human)",
    )
    parser.add_argument(
        "--tvb-dir",
        default="/tmp/tvb",
        help="Path to directory containing TVB zip files (default: /tmp/tvb)",
    )
    args = parser.parse_args(argv)

    for name in ("probability", "amplitude", "axonal_conduction_delay"):
        make_csv_fixture(name, args.cache_dir, SLICE_PARCELS, FIXTURE_DIR)

    for name in ("synaptic_excitatory_delay", "synaptic_inhibitory_delay"):
        make_synaptic_fixture(name, args.cache_dir, SLICE_PARCELS, FIXTURE_DIR)

    make_tvb_signatures(args.tvb_dir, FIXTURE_DIR)


if __name__ == "__main__":
    main()
