"""The first measured human observations in this repository (P19.3).

Every point here was read by hand from a source this host could actually
retrieve, and the manifests say which kind of retrieval that was. Nothing was
taken from a review, a meta-analysis, or another model's digitisation of
somebody else's figure.

What the search produced is not what the task brief expected, so the shape of
this module is worth stating before the data:

* **No per-subject oral-dose-to-striatal-occupancy table exists in reach.**
  The Toronto [11C]raclopride series (Kapur 1996, 1997, 2000) is the right
  study design and every one of the three is paywalled. Their abstracts print
  group ranges and no means, so those points are registered as intervals with
  ``observed_occupancy=None`` rather than as invented midpoints.
* **The two studies whose per-subject tables were readable are each
  disqualified from the oral chain for a different reason.** Xiberas 2001 has
  dose, plasma and striatal values for four patients, and its authors say in
  the text that the striatal values are not occupancy figures, so they are
  registered as ``binding-index``. Videbaek 2001 has four true striatal
  occupancies against measured plasma, and the drug was infused intravenously
  into healthy volunteers and the concentrations are free, not total.
* **The two depot studies (Nyberg 1995, Regenthal 1997) print means, ranges
  and plasma concentrations in their abstracts, and name no tracer.** They are
  registered with ``tracer=None``, which excludes them by construction from
  :func:`~neurocascade.human.pet_validation.observations_by_tracer`.

The fitted relationships that Kapur 1997 and Fitzgerald 2000 report are not
observations and are not stored as such. They are in
:data:`HALOPERIDOL_D2_REPORTED_RELATIONS`, which is what P19.4 should compare
its own fit against.

``docs/p19-3-human-d2-pet-occupancy.md`` has the full search trail, including
what was looked for and not found.
"""
import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

from .dataset_manifest import HumanDatasetManifest, validate_manifest_set
from .pet_validation import PETOccupancyObservation

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"

#: SHA-256 of each committed transcription. These lock our own reading against
#: silent drift. They say nothing about whether the reading matches the paper,
#: which is what ``HumanDatasetManifest.checksum_proves`` spells out.
_FIXTURES = {
    "kapur1996": (
        "haloperidol_d2_kapur1996_raclopride_fixture.csv",
        "62d16e8abf376e6f682ee8e54a40d79c6c952b08ba789cf74456e78e6a54f044",
    ),
    "kapur1997": (
        "haloperidol_d2_kapur1997_raclopride_fixture.csv",
        "fc3881656f85944fd4712067e8581fa28ad9e85a9644c5f5e088c669bb070301",
    ),
    "kapur2000": (
        "haloperidol_d2_kapur2000_raclopride_fixture.csv",
        "0e0042e4d57a0be3337294bca782d8c28fd8de5d1ab731355a2132c768515c32",
    ),
    "nyberg1995": (
        "haloperidol_d2_nyberg1995_decanoate_fixture.csv",
        "6135ad31b34578ce458a02b6765e5d99d78d7c6d51d47478a43b186785b46c6f",
    ),
    "regenthal1997": (
        "haloperidol_d2_regenthal1997_decanoate_fixture.csv",
        "554873cb2a88d7536e8553c58e62a099b5a33c9327a3f03ee7a927ddec461ec8",
    ),
    "videbaek2001": (
        "haloperidol_d2_videbaek2001_ibzm_fixture.csv",
        "989a20d60c15fd24e2bbb088a3eacd4d85969a72ba4e97b67620820899a0f3be",
    ),
    "xiberas2001": (
        "haloperidol_d2_xiberas2001_flb457_fixture.csv",
        "c9f66be6e0881621e16ae88695760552831609f8aed6de3e49d7856e83098273",
    ),
}

_CITATIONS = {
    "kapur1996": (
        "Kapur S, Remington G, Jones C, Wilson A, DaSilva J, Houle S, Zipursky R. "
        "High levels of dopamine D2 receptor occupancy with low-dose haloperidol "
        "treatment: a PET study. Am J Psychiatry 1996;153(7):948-50. "
        "PMID 8659621. doi:10.1176/ajp.153.7.948"
    ),
    "kapur1997": (
        "Kapur S, Zipursky R, Roy P, Jones C, Remington G, Reed K, Houle S. "
        "The relationship between D2 receptor occupancy and plasma levels on low "
        "dose oral haloperidol: a PET study. Psychopharmacology (Berl) "
        "1997;131(2):148-52. PMID 9201802. doi:10.1007/s002130050277"
    ),
    "kapur2000": (
        "Kapur S, Zipursky R, Jones C, Remington G, Houle S. Relationship between "
        "dopamine D2 occupancy, clinical response, and side effects: a double-blind "
        "PET study of first-episode schizophrenia. Am J Psychiatry 2000;157(4):514-20. "
        "PMID 10739409. doi:10.1176/appi.ajp.157.4.514"
    ),
    "nyberg1995": (
        "Nyberg S, Farde L, Halldin C, Dahl ML, Bertilsson L. D2 dopamine receptor "
        "occupancy during low-dose treatment with haloperidol decanoate. "
        "Am J Psychiatry 1995;152(2):173-8. PMID 7840348. doi:10.1176/ajp.152.2.173"
    ),
    "regenthal1997": (
        "Regenthal R, Kunstler U, Junhold U, Hesse S, Sabri O, Preiss R. Haloperidol "
        "serum concentrations and D2 dopamine receptor occupancy during low-dose "
        "treatment with haloperidol decanoate. Int Clin Psychopharmacol "
        "1997;12(5):255-61. PMID 9466159. doi:10.1097/00004850-199709000-00002"
    ),
    "videbaek2001": (
        "Videbaek C, Toska K, Scheideler MA, Paulson OB, Moos Knudsen G. In vivo "
        "measurement of haloperidol affinity to dopamine D2/D3 receptors by "
        "[123I]IBZM and single photon emission computed tomography. "
        "J Cereb Blood Flow Metab 2001;21(1):92-7. PMID 11149673. "
        "doi:10.1097/00004647-200101000-00012"
    ),
    "xiberas2001": (
        "Xiberas X, Martinot JL, Mallet L, Artiges E, Loc'h C, Maziere B, Paillere-"
        "Martinot ML. Extrastriatal and striatal D2 dopamine receptor blockade with "
        "haloperidol or new antipsychotic drugs in patients with schizophrenia. "
        "Br J Psychiatry 2001;179:503-8. PMID 11731352. doi:10.1192/bjp.179.6.503"
    ),
}

_TITLES = {
    "kapur1996": "Haloperidol striatal D2 occupancy, 2 mg/day, [11C]raclopride",
    "kapur1997": "Haloperidol D2 occupancy against plasma level, 1-5 mg/day, [11C]raclopride",
    "kapur2000": "Haloperidol D2 occupancy, first-episode, 1.0-2.5 mg/day, [11C]raclopride",
    "nyberg1995": "Haloperidol decanoate D2 occupancy and plasma at weeks 1 and 4",
    "regenthal1997": "Haloperidol decanoate D2 occupancy and serum at weeks 1 and 4",
    "videbaek2001": "Haloperidol striatal D2 occupancy against free plasma, [123I]IBZM SPECT",
    "xiberas2001": "Haloperidol striatal and extrastriatal binding index, [76Br]FLB 457 PET",
}

#: How each source was obtained. "full text" means a PDF was retrieved and the
#: table read; "abstract" means only the MEDLINE abstract was reachable and the
#: full text is paywalled. Every abstract here was retrieved with efetch.
_ACCESS_ROUTE = {
    "kapur1996": "abstract",
    "kapur1997": "abstract",
    "kapur2000": "abstract",
    "nyberg1995": "abstract",
    "regenthal1997": "abstract",
    "videbaek2001": "full text",
    "xiberas2001": "full text",
}

_RETRIEVAL_DATE = "2026-09-02"


def _optional_float(value):
    value = value.strip()
    return float(value) if value else None


def _optional_int(value):
    value = value.strip()
    return int(value) if value else None


def _optional_str(value):
    value = value.strip()
    return value or None


def _read_fixture(dataset_id):
    filename, expected = _FIXTURES[dataset_id]
    path = FIXTURE_DIR / filename
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected:
        raise ValueError(
            "transcription %s has changed since it was recorded (%s, expected %s); "
            "if the change is a correction, re-read the source and update the "
            "digest deliberately" % (dataset_id, digest, expected)
        )
    source = _CITATIONS[dataset_id]
    observations = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            observations.append(
                PETOccupancyObservation(
                    compound=row["compound"],
                    target=row["target"],
                    dose=_optional_float(row["dose_mg"]),
                    observed_occupancy=_optional_float(row["observed_occupancy"]),
                    source=source,
                    observed_sd=_optional_float(row["observed_sd"]),
                    tracer=_optional_str(row["tracer"]),
                    region=_optional_str(row["region"]),
                    n=_optional_int(row["n"]),
                    quantification_method=_optional_str(row["quantification_method"]),
                    population=_optional_str(row["population"]),
                    dose_regimen=_optional_str(row["dose_regimen"]),
                    plasma_concentration_nM=_optional_float(row["plasma_concentration_nM"]),
                    plasma_concentration_as_printed=_optional_str(
                        row["plasma_concentration_as_printed"]
                    ),
                    plasma_concentration_basis=_optional_str(
                        row["plasma_concentration_basis"]
                    ),
                    source_locator="%s (%s)" % (row["source_locator"], row["subject"]),
                    value_origin=_optional_str(row["value_origin"]),
                    quantity=row["quantity"],
                    observed_range_low=_optional_float(row["observed_range_low"]),
                    observed_range_high=_optional_float(row["observed_range_high"]),
                    notes=row["notes"],
                )
            )
    return tuple(observations)


def haloperidol_d2_manifests():
    """One manifest per source, all transcribed rather than downloaded."""
    manifests = []
    for dataset_id, (filename, digest) in sorted(_FIXTURES.items()):
        manifests.append(
            HumanDatasetManifest(
                dataset_id="haloperidol_d2_%s" % dataset_id,
                layer="pet",
                title=_TITLES[dataset_id],
                source=_CITATIONS[dataset_id],
                citation=_CITATIONS[dataset_id],
                license="not redistributable; values transcribed under fair quotation",
                retrieval_date=_RETRIEVAL_DATE,
                checksum=digest,
                access="open",
                local_path="src/neurocascade/human/fixtures/%s" % filename,
                derivation="transcribed",
                notes="Read from the %s. Values transcribed by hand, not parsed."
                % _ACCESS_ROUTE[dataset_id],
            )
        )
    return validate_manifest_set(manifests)


def haloperidol_d2_observations(dataset_id=None):
    """The registered observations, optionally for one source."""
    if dataset_id is not None:
        if dataset_id not in _FIXTURES:
            raise KeyError("unknown dataset_id: %s" % dataset_id)
        return _read_fixture(dataset_id)
    collected = []
    for key in sorted(_FIXTURES):
        collected.extend(_read_fixture(key))
    return tuple(collected)


def occupancy_observations(observations=None):
    """Only the points that are occupancies, and only those with a value.

    Binding indices, availability ratios and interval-only points are dropped,
    because they are the three ways a caller can accidentally treat something
    that is not a usable occupancy as if it were one.
    """
    if observations is None:
        observations = haloperidol_d2_observations()
    return tuple(
        item
        for item in observations
        if item.quantity == "occupancy" and item.observed_occupancy is not None
    )


@dataclass(frozen=True)
class ReportedOccupancyRelation:
    """A plasma-concentration to occupancy relation as its authors reported it.

    This is a fitted result, not an observation, and it is kept apart from the
    observation set so that a model fitted in P19.4 is not silently compared
    against something that already is a model.
    """

    label: str
    compound: str
    target: str
    tracer: str
    region: str | None
    n: int
    model: str
    ec50_ng_per_mL: float
    ec50_as_printed: str
    plasma_concentration_basis: str
    source: str
    source_locator: str
    value_origin: str
    fit_quality: str | None = None
    notes: str = ""

    def __post_init__(self):
        if self.ec50_ng_per_mL <= 0:
            raise ValueError("ec50_ng_per_mL must be positive")
        if self.n < 1:
            raise ValueError("n must be at least 1")
        if self.plasma_concentration_basis not in {"total", "free"}:
            raise ValueError("plasma_concentration_basis must be total or free")

    @property
    def ec50_nM(self):
        from .pet_validation import HALOPERIDOL_MW_G_PER_MOL

        return 1000.0 * self.ec50_ng_per_mL / HALOPERIDOL_MW_G_PER_MOL

    def occupancy_at(self, plasma_ng_per_mL):
        """Occupancy from the authors' own rectangular hyperbola."""
        if plasma_ng_per_mL < 0:
            raise ValueError("plasma concentration must not be negative")
        return plasma_ng_per_mL / (plasma_ng_per_mL + self.ec50_ng_per_mL)


#: The two published plasma-to-occupancy relations for haloperidol that this
#: host could read. They disagree, and the disagreement is between a paper and
#: the paper that validated it, so it is not averaged away here.
HALOPERIDOL_D2_REPORTED_RELATIONS = (
    ReportedOccupancyRelation(
        label="kapur1997",
        compound="haloperidol",
        target="D2",
        tracer="[11C]raclopride",
        region=None,
        n=12,
        model="rectangular hyperbola, occupancy = C / (C + EC50)",
        ec50_ng_per_mL=0.51,
        ec50_as_printed="50% D2 occupancy was achieved with 0.51 ng/ml",
        plasma_concentration_basis="total",
        source=_CITATIONS["kapur1997"],
        source_locator="abstract, Results",
        value_origin="text",
        fit_quality="r2 = 0.84",
        notes=(
            "The abstract also prints 80% occupancy at 2.0 ng/ml. A single "
            "hyperbola with EC50 0.51 ng/ml gives 2.04, so the two printed "
            "points are consistent with one another and with this EC50."
        ),
    ),
    ReportedOccupancyRelation(
        label="fitzgerald2000",
        compound="haloperidol",
        target="D2",
        tracer="[11C]raclopride",
        region="striatum",
        n=21,
        model="rectangular hyperbola, occupancy = C / (C + ED50)",
        ec50_ng_per_mL=0.40,
        ec50_as_printed=(
            "%D2 occupancy=plasma level/ED50+plasma level, where ED50= 0.40 ng/ml"
        ),
        plasma_concentration_basis="total",
        source=(
            "Fitzgerald PB, Kapur S, Remington G, Roy P, Zipursky RB. Predicting "
            "haloperidol occupancy of central dopamine D2 receptors from plasma "
            "levels. Psychopharmacology (Berl) 2000;149(1):1-5. PMID 10789875. "
            "doi:10.1007/s002139900333"
        ),
        source_locator="abstract, Methods and Results",
        value_origin="text",
        fit_quality="prospective test: mean error 3.89% (CI 0.45-7.33), Pearson r = 0.864, P = 0.003",
        notes=(
            "This paper describes 0.40 ng/ml as the previously derived ED50 and "
            "tests it prospectively in 21 new patients. Kapur 1997's own abstract "
            "implies 0.51 ng/ml, so the validation paper restates the derivation "
            "paper with a different constant. Both are recorded; neither is "
            "averaged into the other."
        ),
    ),
)
