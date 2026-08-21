"""P3.1: the Allen CCF structure ontology, as a hierarchy you can ask
questions of.

Why the connectome needs this at all
------------------------------------

Knox et al. 2019's matrix is indexed by bare integer structure ids (184,
985, 993, ...). Oh et al. 2014's is indexed by bare acronyms (FRP, MOp,
ACAd, ...). Neither is usable on its own:

* nothing tells you that id 749 is the ventral tegmental area, which is
  the node this codebase's opioid scenario already models;
* nothing tells you that ids 985 and 993 are both isocortex while 749 is
  midbrain, which is what P3.2's "selectable resolution" means in
  practice - a coarse network is the fine network aggregated up the
  hierarchy;
* nothing lets the two datasets be compared, because one speaks ids and
  the other acronyms.

The structure graph supplies all three: a tree of 1327 structures, each
with an id, an acronym, a full name, and a parent, rooted at `root`
(id 997).

Design decisions worth recording
--------------------------------

**The tree is flattened on load, and the hierarchy is kept as parent
pointers.** The source JSON is nested (`children` lists). Every query this
project actually makes - "what division is this region in", "is A inside
B", "give me the leaves under B" - is either an upward walk or a downward
walk, and parent pointers plus a children index serve both without keeping
the raw nesting around. `depth` is computed on load rather than trusting
`st_level`, which is present in the source but is an atlas-drawing level,
not a graph depth, and is null for some structures.

**Acronyms are treated as unique keys, but only after checking.** All 1327
acronyms in graph 1 are distinct (verified on load; `by_acronym` raises if
that ever stops being true rather than silently keeping the last one).
That is what makes the Oh 2014 acronym-indexed matrix joinable to the
Knox id-indexed one at all.

**The twelve major divisions are not hard-coded from memory.** They are
the twelve structure ids that Knox et al. 2019 themselves used as the
source/target mask for the regionalized model, published alongside the
matrices as `source_mask_params.json` / `target_mask_params.json`:
Isocortex, OLF, HPF, CTXsp, STR, PAL, TH, HY, MB, P, MY, CB. The count
matches that paper's Methods ("The brain is divided into a set of s = 12
major brain divisions at a high level of the CCF ontology"), so the mask
files and the prose agree. Recording them here as ids, with the acronyms
resolved from the ontology at runtime, means a rename upstream shows up as
a resolution failure rather than as a silently wrong grouping.
"""

import json

#: The twelve major brain divisions spanned by the Knox et al. 2019
#: regionalized model, taken from that publication's own published
#: `source_mask_params.json` / `target_mask_params.json` (both list exactly
#: these ids). Order is the published order.
MAJOR_DIVISION_IDS = (
    315,   # Isocortex
    698,   # OLF   - olfactory areas
    1089,  # HPF   - hippocampal formation
    703,   # CTXsp - cortical subplate
    477,   # STR   - striatum
    803,   # PAL   - pallidum
    549,   # TH    - thalamus
    1097,  # HY    - hypothalamus
    313,   # MB    - midbrain
    771,   # P     - pons
    354,   # MY    - medulla
    512,   # CB    - cerebellum
)


class Structure(object):
    """One node of the CCF ontology.

    `depth` is the number of edges to `root`, computed on load. `parent_id`
    is None only for `root` itself.
    """

    __slots__ = (
        "id",
        "acronym",
        "name",
        "parent_id",
        "st_level",
        "graph_order",
        "color_hex_triplet",
        "depth",
    )

    def __init__(self, id, acronym, name, parent_id, st_level, graph_order,
                 color_hex_triplet, depth):
        self.id = id
        self.acronym = acronym
        self.name = name
        self.parent_id = parent_id
        self.st_level = st_level
        self.graph_order = graph_order
        self.color_hex_triplet = color_hex_triplet
        self.depth = depth

    def __repr__(self):
        return "Structure(%d, %r, %r)" % (self.id, self.acronym, self.name)


class StructureOntology(object):
    """The flattened CCF structure graph, with upward and downward queries."""

    def __init__(self, structures):
        self.structures = {s.id: s for s in structures}
        if len(self.structures) != len(structures):
            raise ValueError("duplicate structure ids in ontology")
        acronyms = {}
        for structure in structures:
            if structure.acronym in acronyms:
                raise ValueError(
                    "duplicate acronym %r (ids %d and %d): this ontology "
                    "cannot be keyed by acronym, and the Oh 2014 matrix "
                    "loader depends on that"
                    % (structure.acronym, acronyms[structure.acronym], structure.id)
                )
            acronyms[structure.acronym] = structure.id
        self._by_acronym = acronyms
        children = {}
        for structure in structures:
            if structure.parent_id is not None:
                children.setdefault(structure.parent_id, []).append(structure.id)
        self._children = children

    # --- construction ------------------------------------------------------

    @classmethod
    def from_json(cls, payload):
        """Build from the parsed `structure_graph_download/1.json` payload.

        Accepts either the full API envelope ({"msg": [root]}) or a bare
        root node, because the committed test fixture is a trimmed tree in
        the same nested shape and it is not worth two formats.
        """
        if isinstance(payload, dict) and "msg" in payload:
            roots = payload["msg"]
        elif isinstance(payload, list):
            roots = payload
        else:
            roots = [payload]

        structures = []

        def walk(node, depth):
            structures.append(
                Structure(
                    id=int(node["id"]),
                    acronym=node["acronym"],
                    name=node["name"],
                    parent_id=(
                        None
                        if node.get("parent_structure_id") is None
                        else int(node["parent_structure_id"])
                    ),
                    st_level=node.get("st_level"),
                    graph_order=node.get("graph_order"),
                    color_hex_triplet=node.get("color_hex_triplet"),
                    depth=depth,
                )
            )
            for child in node.get("children", []):
                walk(child, depth + 1)

        for root in roots:
            walk(root, 0)
        return cls(structures)

    @classmethod
    def from_path(cls, path):
        with open(path, "r") as handle:
            return cls.from_json(json.load(handle))

    # --- lookup ------------------------------------------------------------

    def __len__(self):
        return len(self.structures)

    def __contains__(self, structure_id):
        return int(structure_id) in self.structures

    def __getitem__(self, structure_id):
        try:
            return self.structures[int(structure_id)]
        except KeyError:
            raise KeyError("structure id %r is not in this ontology" % (structure_id,))

    def by_acronym(self, acronym):
        """Structure with this acronym. Raises KeyError with the acronym in it."""
        try:
            return self.structures[self._by_acronym[acronym]]
        except KeyError:
            raise KeyError("acronym %r is not in this ontology" % (acronym,))

    def id_for_acronym(self, acronym):
        return self.by_acronym(acronym).id

    def acronym_for_id(self, structure_id):
        return self[structure_id].acronym

    def name_for_id(self, structure_id):
        return self[structure_id].name

    # --- hierarchy ---------------------------------------------------------

    def ancestors(self, structure_id, include_self=False):
        """Ids from `structure_id` upward to the root.

        Ordered nearest-first (parent, grandparent, ..., root), because
        every use here is "the first ancestor that is also X".
        """
        out = []
        current = self[structure_id]
        if include_self:
            out.append(current.id)
        while current.parent_id is not None:
            current = self[current.parent_id]
            out.append(current.id)
        return out

    def children(self, structure_id):
        return list(self._children.get(int(structure_id), []))

    def descendants(self, structure_id, include_self=False):
        """Every id in the subtree below `structure_id`, pre-order."""
        out = []
        stack = [int(structure_id)]
        if not include_self:
            stack = self.children(structure_id)
        while stack:
            current = stack.pop()
            out.append(current)
            stack.extend(reversed(self.children(current)))
        return out

    def is_descendant_of(self, structure_id, ancestor_id):
        return int(ancestor_id) in self.ancestors(structure_id, include_self=True)

    def ancestor_at(self, structure_id, candidate_ids):
        """The first id in `candidate_ids` on the path from this structure up.

        Returns None when the structure sits outside every candidate
        subtree. This is the single primitive P3.2's "selectable
        resolution" is built from: aggregate to a coarse parcellation by
        mapping every fine region to its ancestor in the coarse set.
        """
        candidates = set(int(c) for c in candidate_ids)
        for current in self.ancestors(structure_id, include_self=True):
            if current in candidates:
                return current
        return None

    def major_division(self, structure_id):
        """Which of the twelve major divisions contains this structure.

        Returns the division's structure id, or None for structures outside
        all twelve (root, `grey`, fibre tracts, ventricles).
        """
        return self.ancestor_at(structure_id, MAJOR_DIVISION_IDS)

    def major_divisions(self):
        """The twelve divisions as Structure objects, in published order.

        Raises if any id is missing, which is the intended behaviour: a
        division that cannot be resolved means this ontology is not the one
        the connectome was built against.
        """
        return [self[division_id] for division_id in MAJOR_DIVISION_IDS]

    def path_acronyms(self, structure_id):
        """Root-first acronym path, e.g. ['root', 'grey', 'BS', ...,'VTA']."""
        return [
            self[i].acronym
            for i in reversed(self.ancestors(structure_id, include_self=True))
        ]


def load_ontology(path=None, cache_dir=None, allow_download=False):
    """Load the CCF ontology from the verified cache (or an explicit path).

    `allow_download` defaults to False so that no ordinary import or test
    can start a download by accident; the fetch script passes True.
    """
    if path is None:
        from . import fetch
        from .sources import ONTOLOGY_KEY

        path = fetch.ensure(
            ONTOLOGY_KEY, cache_dir=cache_dir, allow_download=allow_download
        )
    return StructureOntology.from_path(path)
