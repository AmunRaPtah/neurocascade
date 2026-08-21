"""P3.2: region voxel counts, recovered from the published matrices.

Why this module has to exist
----------------------------

P3.2 has to build a network at three resolutions, which means aggregating
fine regions into coarse ones. Aggregating a *normalized* quantity is not
a sum. Writing `s_ij` for the summed voxel-pair model weight from source
region `i` to target region `j`, and `|X_i|`, `|Y_j|` for the two regions'
voxel counts, the four published Knox et al. 2019 matrices are

    connection_strength             s_ij
    connection_density              s_ij / |Y_j|
    normalized_connection_strength  s_ij / |X_i|
    normalized_connection_density   s_ij / (|X_i| |Y_j|)

Only the first is additive. The density of a coarse region pair (I, J) is

    (sum over i in I, j in J of s_ij) / (sum over j in J of |Y_j|),

i.e. exactly what Knox et al. would have published had they regionalized
at that coarser level - and it needs the voxel counts. Averaging the
children's densities instead would weight a 483-voxel nucleus the same as
a 6,544-voxel cortical area, which for the mouse brain is a factor of
thirteen wrong on a single pair.

Where the numbers come from: nowhere new
----------------------------------------

The voxel counts are not fetched, and they are not written down anywhere in
this codebase. They are *recovered* from the ratios of matrices already in
P3.1's manifest:

    connection_strength / connection_density                  = |Y_j|
    normalized_connection_strength / normalized_connection_density = |Y_j|
    connection_density / normalized_connection_density        = |X_i|
    connection_strength / normalized_connection_strength      = |X_i|

so no new source, no new checksum, and no new number enters the project.
On the published 291 x 577 files each of those four ratios is constant
along its axis to 2.2e-16 (one ulp) and every recovered value is an exact
integer, which is a strong check that the normalizations really are what
the manifest says they are - it is the same fact derived four ways.

This module therefore also serves as an independent test of P3.1's `units`
strings: if Allen ever re-published `connection_density.csv.gz` with the
source size divided out instead of the target size, the constancy check
below would fail loudly rather than producing a plausible wrong network.

Units
-----

The recovered numbers are **voxel counts in the CCF grid the Knox et al.
2019 regionalized model was built on** (100 um isotropic; the 291
ipsilateral regions sum to 209,134 voxels = 209.1 mm^3, which is the right
order for one hemisphere of grey matter in a mouse). Nothing here converts
them to mm^3, because nothing downstream needs a physical volume: they are
used only as weights in an aggregation, where the grid constant cancels.
"""

import numpy as np

#: Relative spread tolerated along the constant axis. This is NOT a fitted
#: tolerance: on the published files the observed spread is 2.2e-16 (one
#: double-precision ulp), and this bound is a few ulp of headroom for the
#: division. A real normalization mismatch is off by orders of magnitude,
#: not by 1e-9, so the check is not made weak by the headroom.
_RATIO_RTOL = 1e-9

#: How close a recovered count must be to a whole number. The published
#: files give exact integers; this allows for the last bit of a division.
_INTEGER_ATOL = 1e-6


def _paired(numerator, denominator):
    if numerator.weights.shape != denominator.weights.shape:
        raise ValueError(
            "%s and %s have different shapes (%s vs %s)"
            % (
                numerator.dataset_key,
                denominator.dataset_key,
                numerator.weights.shape,
                denominator.weights.shape,
            )
        )
    if not np.array_equal(numerator.source_ids, denominator.source_ids):
        raise ValueError(
            "%s and %s do not have the same source regions in the same order"
            % (numerator.dataset_key, denominator.dataset_key)
        )
    if not np.array_equal(numerator.target_ids, denominator.target_ids):
        raise ValueError(
            "%s and %s do not have the same target columns in the same order"
            % (numerator.dataset_key, denominator.dataset_key)
        )
    if list(numerator.target_hemispheres) != list(denominator.target_hemispheres):
        raise ValueError(
            "%s and %s disagree about which columns are contralateral"
            % (numerator.dataset_key, denominator.dataset_key)
        )
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = numerator.weights / denominator.weights
    usable = np.isfinite(ratio) & (ratio > 0.0)
    return ratio, usable


def _constant_along(ratio, usable, axis, labels, what):
    """Collapse `ratio` along `axis`, checking it really is constant there.

    `axis=0` collapses rows (one value per column), `axis=1` collapses
    columns (one value per row).
    """
    n = ratio.shape[1 - axis]
    out = np.empty(n, dtype=float)
    for k in range(n):
        column = ratio[:, k] if axis == 0 else ratio[k, :]
        mask = usable[:, k] if axis == 0 else usable[k, :]
        values = column[mask]
        if values.size == 0:
            raise ValueError(
                "cannot recover %s for %s: every entry of that %s is zero or "
                "non-finite in at least one of the two matrices"
                % (what, labels[k], "column" if axis == 0 else "row")
            )
        reference = float(np.median(values))
        spread = float(np.max(np.abs(values / reference - 1.0)))
        if spread > _RATIO_RTOL:
            raise ValueError(
                "the ratio of the two matrices is not constant along %s for "
                "%s (relative spread %.3g > %.3g), so they are not the two "
                "normalizations this recovery assumes; check the manifest's "
                "units strings before trusting either matrix"
                % ("that column" if axis == 0 else "that row", labels[k], spread, _RATIO_RTOL)
            )
        out[k] = reference
    off = np.abs(out - np.round(out))
    if np.any(off > _INTEGER_ATOL):
        worst = int(np.argmax(off))
        raise ValueError(
            "recovered %s for %s is %.10g, which is not a whole number of "
            "voxels; the two matrices are not related by a voxel-count "
            "normalization" % (what, labels[worst], out[worst])
        )
    return np.round(out)


def target_voxel_counts(strength, density):
    """Voxel count of every target column, as `{(hemisphere, id): count}`.

    `strength` must be `connection_strength` (raw `s_ij`) and `density`
    `connection_density` (`s_ij / |Y_j|`) over the same regions in the same
    order; the pair `normalized_connection_strength` /
    `normalized_connection_density` works identically and is accepted.
    """
    ratio, usable = _paired(strength, density)
    labels = [
        "%d (%s)" % (int(i), h)
        for i, h in zip(strength.target_ids, strength.target_hemispheres)
    ]
    counts = _constant_along(ratio, usable, axis=0, labels=labels, what="target voxel count")
    return {
        (str(h), int(i)): float(c)
        for h, i, c in zip(strength.target_hemispheres, strength.target_ids, counts)
    }


def source_voxel_counts(strength, normalized_strength):
    """Voxel count of every source row, as `{id: count}`.

    `strength` must be `connection_strength` and `normalized_strength`
    `normalized_connection_strength` (`s_ij / |X_i|`); the pair
    `connection_density` / `normalized_connection_density` works identically.
    """
    ratio, usable = _paired(strength, normalized_strength)
    labels = ["%d" % int(i) for i in strength.source_ids]
    counts = _constant_along(
        ratio, usable, axis=1, labels=labels, what="source voxel count"
    )
    return {int(i): float(c) for i, c in zip(strength.source_ids, counts)}
