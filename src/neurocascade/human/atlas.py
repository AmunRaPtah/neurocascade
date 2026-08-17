"""The Desikan-Killiany human atlas: 68 cortical + 14 subcortical regions.

The canonical region identifier in this package is the ENIGMA Toolbox's
label (`L_bankssts`, `R_precentral`, `Laccumb`, `Rthal`, ...). `parse_label`
turns a label into (hemisphere, suffix, kind); `atlas_from_labels` turns
the label list that ships alongside the matrix into a `HumanAtlas` of
`Region` objects carrying the full Desikan-Killiany region name.

The full names are the standard Desikan-Killiany names (Desikan et al.
2006); the subcortical names are the classic nuclei. They are written down
here so a built network can print a human-readable region name, not just
the file's compact label.
"""

DK_CORTICAL_NAMES = {
    "bankssts": "banks of the superior temporal sulcus",
    "caudalanteriorcingulate": "caudal anterior cingulate cortex",
    "caudalmiddlefrontal": "caudal middle frontal gyrus",
    "cuneus": "cuneus",
    "entorhinal": "entorhinal cortex",
    "fusiform": "fusiform gyrus",
    "inferiorparietal": "inferior parietal cortex",
    "inferiortemporal": "inferior temporal gyrus",
    "isthmuscingulate": "isthmus of the cingulate cortex",
    "lateraloccipital": "lateral occipital cortex",
    "lateralorbitofrontal": "lateral orbitofrontal cortex",
    "lingual": "lingual gyrus",
    "medialorbitofrontal": "medial orbitofrontal cortex",
    "middletemporal": "middle temporal gyrus",
    "parahippocampal": "parahippocampal gyrus",
    "paracentral": "paracentral lobule",
    "parsopercularis": "pars opercularis",
    "parsorbitalis": "pars orbitalis",
    "parstriangularis": "pars triangularis",
    "pericalcarine": "pericalcarine cortex",
    "postcentral": "postcentral gyrus",
    "posteriorcingulate": "posterior cingulate cortex",
    "precentral": "precentral gyrus",
    "precuneus": "precuneus",
    "rostralanteriorcingulate": "rostral anterior cingulate cortex",
    "rostralmiddlefrontal": "rostral middle frontal gyrus",
    "superiorfrontal": "superior frontal gyrus",
    "superiorparietal": "superior parietal cortex",
    "superiortemporal": "superior temporal gyrus",
    "supramarginal": "supramarginal gyrus",
    "frontalpole": "frontal pole",
    "temporalpole": "temporal pole",
    "transversetemporal": "transverse temporal cortex",
    "insula": "insula",
}

SUBCORTICAL_NAMES = {
    "accumb": "nucleus accumbens",
    "amyg": "amygdala",
    "caud": "caudate nucleus",
    "hippo": "hippocampus",
    "pal": "pallidum",
    "put": "putamen",
    "thal": "thalamus",
}

KINDS = ("cortex", "subcortex")
HEMISPHERES = ("L", "R")


def parse_label(label):
    """Split an ENIGMA label into (hemisphere, suffix, kind).

    Cortical labels carry a hemisphere underscore prefix (`L_`, `R_`);
    subcortical labels carry the hemisphere letter directly (`L`, `R`).
    A label that matches neither is an error, not a silently-unknown region.
    """
    if label.startswith("L_"):
        return "L", label[2:], "cortex"
    if label.startswith("R_"):
        return "R", label[2:], "cortex"
    if label.startswith("L"):
        return "L", label[1:], "subcortex"
    if label.startswith("R"):
        return "R", label[1:], "subcortex"
    raise ValueError(
        "cannot parse ENIGMA region label %r: expected a cortical "
        "'L_*'/'R_*' or subcortical 'L*'/'R*' label" % label
    )


def _full_name(suffix, kind):
    table = DK_CORTICAL_NAMES if kind == "cortex" else SUBCORTICAL_NAMES
    try:
        return table[suffix]
    except KeyError:
        raise ValueError(
            "unknown %s region suffix %r: not a Desikan-Killiany cortical "
            "region or a recognized subcortical nucleus" % (kind, suffix)
        )


class Region(object):
    """One atlas region: the file's label plus its parsed identity."""

    def __init__(self, id, label, hemisphere, kind, suffix, name):
        self.id = id
        self.label = label
        self.hemisphere = hemisphere
        self.kind = kind
        self.suffix = suffix
        self.name = name

    def __repr__(self):
        return "Region(%r, %s, %s)" % (self.label, self.kind, self.hemisphere)


class HumanAtlas(object):
    """The ordered region set behind one connectome matrix."""

    def __init__(self, regions):
        self.regions = list(regions)
        self.by_label = {r.label: r for r in self.regions}
        self._by_id = {r.id: r for r in self.regions}
        if len(self.by_label) != len(self.regions):
            raise ValueError("duplicate region label in atlas")
        if len(self._by_id) != len(self.regions):
            raise ValueError("duplicate region id in atlas")

    @property
    def labels(self):
        return [r.label for r in self.regions]

    def __len__(self):
        return len(self.regions)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._by_id[key]
        return self.by_label[key]

    def cortical(self):
        return [r for r in self.regions if r.kind == "cortex"]

    def subcortical(self):
        return [r for r in self.regions if r.kind == "subcortex"]

    def hemisphere_counts(self):
        """{(hemisphere, kind): count} - what a test asserts on."""
        counts = {}
        for region in self.regions:
            key = (region.hemisphere, region.kind)
            counts[key] = counts.get(key, 0) + 1
        return counts


def atlas_from_labels(labels):
    """Build a `HumanAtlas` from an ordered list of ENIGMA labels."""
    regions = []
    for i, label in enumerate(labels):
        hemisphere, suffix, kind = parse_label(label)
        regions.append(
            Region(
                id=i,
                label=label,
                hemisphere=hemisphere,
                kind=kind,
                suffix=suffix,
                name=_full_name(suffix, kind),
            )
        )
    return HumanAtlas(regions)
