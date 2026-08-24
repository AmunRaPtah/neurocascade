"""decoder.py's five graded drive-response series, retired here by P11.3.

Between P7.2 and P11.3 these five `DriveResponseSeries` objects were
`decoder.SERIES`: the entire graded single-region library the nonlinear
decoder assessed itself against. Every one describes a MOUSE or RAT experiment
(four mouse optogenetic series from Herent 2023 / Caggiano 2018 / De Luca
2022, one rat microinjection series from Diaz 2011), so under the v2
human-primary roadmap none of them may inform a human readout, and P11.3
moved them here whole.

What stayed in production is everything else about `decoder`: the
`DriveResponseSeries` type, the four fit families, leave-one-out machinery,
verdict rules and `SEARCHED_AND_EMPTY`. Its library is now empty
(`decoder.SERIES == ()`), every endpoint reports `no-data`, and nothing was
ever adopted - which was also true when the five series were present. The
machinery runs against synthetic series in `tests/test_decoder.py`; this
module keeps the real rodent series importable and tested in
`tests/test_legacy_rodent_series.py`.

The five verdicts are preserved as they were on the day of retirement:
herent2023 -> weighting-dependent, caggiano cnf -> no-signal,
caggiano ppn -> linear-wins, deluca2022 -> saturating-wins-not-identifiable,
diaz2011 -> too-few-levels. None licensed a nonlinearity, so retiring them
changed no prediction.

The citation KEYS these series name resolve in `decoder.CITATIONS`, which
keeps all fifteen bibliographic entries - including the four below that are
now cited only in prose there - so there is exactly one copy of each
reference.
"""

from .. import decoder

#: The five series exactly as `decoder.SERIES` held them until P11.3.
SERIES = (
    decoder.DriveResponseSeries(
        key="herent2023_cnf_locomotion",
        endpoint="locomotion",
        structures=("CUN",),
        manipulation="optogenetic_stimulation",
        drive_units="Hz",
        levels=(5.0, 10.0, 15.0, 20.0, 30.0, 40.0),
        values=(0.187214, 0.128781, 54.3845, 67.4137, 84.8552, 85.9764),
        value_units="mean displacement speed, cm/s",
        score_sign=1,
        dispersion=(0.352991, 0.115215, 20.8858, 10.7764, 18.6683, 13.4384),
        dispersion_kind="SD",
        n_per_level=(5, 5, 5, 5, 5, 5),
        citation="Herent2023",
        species="mouse",
        provenance="source data",
        finding=(
            "Photoactivating glutamatergic cuneiform neurons at 15 Hz or "
            "more engages forward locomotion and higher frequencies impose "
            "faster regimes; 5 and 10 Hz are below the threshold for "
            "locomotor initiation and evoke essentially none."
        ),
        note=(
            "The 'Mean' and 'SD' rows are the file's own, over the five "
            "per-animal means; the sheet also holds all 15 individual "
            "trials per frequency. The SDs at 5 and 10 Hz (0.35 and 0.12 "
            "cm/s) are a FLOOR, not precision - those frequencies evoke no "
            "locomotion in any animal - which is why this series is the "
            "reason the module assesses under both weightings."
        ),
    ),
    decoder.DriveResponseSeries(
        key="caggiano2018_cnf_locomotion",
        endpoint="locomotion",
        structures=("CUN",),
        manipulation="optogenetic_stimulation",
        drive_units="Hz",
        levels=(5.0, 10.0, 20.0, 50.0),
        values=(23.6770, 30.3664, 50.5043, 57.2240),
        value_units="mean of per-trial maximum speed, cm/s",
        score_sign=1,
        dispersion=(7.6196, 5.2498, 6.2010, 4.5634),
        dispersion_kind="SEM",
        n_per_level=(14, 41, 37, 39),
        citation="Caggiano2018",
        species="mouse",
        provenance="source data",
        finding=(
            "Stepwise increasing the stimulation frequency from threshold "
            "values around 5 Hz to maximal frequencies at 50 Hz "
            "progressively increased the speed of locomotion."
        ),
        note=(
            "Means and SEMs COMPUTED HERE from the deposit's trial-level "
            "maximum speeds; the paper plots medians with 25th/75th "
            "percentiles and states no per-level number. Two discrepancies "
            "in the deposit are recorded rather than smoothed. (1) The "
            "level column is headed 'Power', but its values are "
            "{5, 10, 20, 50} while the paper's stated laser power range is "
            "5-30 mW, and the Results sentence quoted above names 5 Hz and "
            "50 Hz as the ends of a FREQUENCY series - so the column is "
            "frequency in Hz and its header is wrong. (2) The deposit holds "
            "131 CnF trials where the figure legend says n = 148; the n "
            "used here is the deposit's."
        ),
    ),
    decoder.DriveResponseSeries(
        key="caggiano2018_ppn_locomotion",
        endpoint="locomotion",
        structures=("PPN",),
        manipulation="optogenetic_stimulation",
        drive_units="Hz",
        levels=(5.0, 10.0, 20.0, 50.0),
        values=(1.6416, 5.4990, 5.8049, 19.8448),
        value_units="mean of per-trial maximum speed, cm/s",
        score_sign=1,
        dispersion=(0.5351, 2.4202, 3.3058, 5.8237),
        dispersion_kind="SEM",
        n_per_level=(4, 21, 18, 24),
        citation="Caggiano2018",
        species="mouse",
        provenance="source data",
        finding=(
            "The same frequency series delivered to glutamatergic "
            "pedunculopontine neurons evokes far slower locomotion than the "
            "cuneiform nucleus does (pooled maxima 19 vs 56 cm/s)."
        ),
        note=(
            "Same deposit and same two caveats as the CnF series. The 5 Hz "
            "level rests on four trials, which the leave-one-out numbers "
            "reflect rather than hide."
        ),
    ),
    decoder.DriveResponseSeries(
        key="deluca2022_vlpo_sedation",
        endpoint="sedation",
        structures=("VLPO",),
        manipulation="optogenetic_stimulation",
        drive_units="Hz",
        levels=(0.0, 1.0, 5.0, 10.0, 20.0),
        values=(32.242, 33.701, 48.206, 53.854, 59.234),
        value_units="probability of arousal from NREM sleep, %",
        score_sign=-1,
        dispersion=(3.064, 2.5149, 4.1048, 6.0107, 11.206),
        dispersion_kind="SEM",
        n_per_level=(5, 5, 5, 5, 5),
        citation="DeLuca2022",
        species="mouse",
        provenance="source data",
        finding=(
            "Arousal probability from NREM sleep rose with the frequency of "
            "orexinergic stimulation in the VLPO, significantly at 10 and "
            "20 Hz and not at 1 or 5 Hz."
        ),
        note=(
            "The zero level is the paper's own sham. `score_sign=-1` "
            "because the sedation endpoint's direction is 'higher = MORE "
            "sedated', and a higher arousal probability is less sedation. "
            "This is a TERMINAL-FIELD manipulation: ChR2 sits in lateral "
            "hypothalamic orexin neurons and the fibre sits over the VLPO, "
            "so raising the drive lowers the VLPO sleep-active rate. That "
            "is monotone DECREASING, which the identifiability argument "
            "accommodates - it needs monotone, not increasing - and the "
            "region whose rate is graded is unambiguously the VLPO."
        ),
    ),
    decoder.DriveResponseSeries(
        key="diaz2011_bla_anxiety",
        endpoint="anxiety",
        structures=("BLA",),
        manipulation="intracerebral_microinjection",
        drive_units="pmol",
        levels=(0.0, 1.5, 5.0),
        values=(7.5, 41.7, 104.8),
        value_units="open-arm time, s of 300",
        score_sign=-1,
        dispersion=(3.7, 5.5, 16.9),
        dispersion_kind="SEM",
        citation="Diaz2011",
        species="rat",
        provenance="table",
        finding=(
            "Bilateral intra-BLA U99194 increased open-arm time from "
            "7.5 s at vehicle to 41.7 s at 1.5 pmol and 104.8 s at "
            "5.0 pmol of a 300 s trial, with matching increases in open-arm "
            "entries and percentage of open-arm entries."
        ),
        note=(
            "Three levels, so it is rejected as `too-few-levels` before any "
            "family is compared - it is here because a graded intra-BLA "
            "series with per-dose EPM means in a TABLE is close to unique, "
            "and because a rejection for want of a fourth level is a more "
            "useful record than silence. Table 1 does not restate n; the "
            "paper's light/dark text gives 6, 8 and 6 for the three groups "
            "and is internally inconsistent about the middle one, so "
            "`n_per_level` is left unset and the dispersion weighting uses "
            "the published SEMs directly. Open, closed and junction times "
            "sum to 300.1, 300.1 and 299.9 s, which is the only internal "
            "check the table supports and it passes."
        ),
    ),
)

__all__ = ["SERIES"]
