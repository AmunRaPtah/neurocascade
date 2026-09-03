"""Provenance-first dataset manifests for human validation data.

The manifest describes data without bundling restricted clinical records.  It
also provides small adapters into the existing PET, PK, and behavioral
contracts once a caller has independently reviewed the source data.
"""
from dataclasses import dataclass
from datetime import date
from typing import Literal


Layer = Literal["pk", "bbb", "pet", "dose_response", "heldout_behavior"]

#: How the local artefact came to exist. This matters because a checksum
#: proves different things in each case. For a downloaded file the checksum
#: ties the local copy to a published artefact. For a transcribed or digitised
#: table there is no upstream file: the checksum only locks our own reading
#: against later drift, which is worth having and is not evidence of fidelity
#: to the paper. Saying which is which stops the second being read as the
#: first.
Derivation = Literal["downloaded", "transcribed", "digitised"]


@dataclass(frozen=True)
class HumanDatasetManifest:
    dataset_id: str
    layer: Layer
    title: str
    source: str
    citation: str
    license: str
    retrieval_date: str
    checksum: str | None = None
    access: Literal["open", "controlled", "author_request"] = "open"
    local_path: str | None = None
    notes: str = ""
    derivation: Derivation = "downloaded"

    def __post_init__(self):
        if self.derivation not in {"downloaded", "transcribed", "digitised"}:
            raise ValueError("invalid derivation")
        if not self.dataset_id or not self.title or not self.source or not self.citation:
            raise ValueError("dataset_id, title, source, and citation are required")
        if self.layer not in {"pk", "bbb", "pet", "dose_response", "heldout_behavior"}:
            raise ValueError("unsupported human validation layer")
        if self.access not in {"open", "controlled", "author_request"}:
            raise ValueError("invalid dataset access status")
        try:
            date.fromisoformat(self.retrieval_date)
        except ValueError as exc:
            raise ValueError("retrieval_date must be ISO-8601 YYYY-MM-DD") from exc
        if self.checksum is not None and len(self.checksum) != 64:
            raise ValueError("checksum must be a SHA-256 hex digest")
        if self.checksum is not None:
            int(self.checksum, 16)

    @property
    def redistributable(self) -> bool:
        return self.access == "open" and bool(self.local_path)

    @property
    def checksum_proves(self) -> str:
        """What this manifest's checksum does and does not establish."""
        if self.checksum is None:
            return "nothing: no checksum recorded"
        if self.derivation == "downloaded":
            return "that the local copy is byte-identical to the published artefact"
        return (
            "only that our %s reading has not drifted since it was recorded; "
            "fidelity to the source has to be checked by reading the source"
            % self.derivation
        )


def validate_manifest(manifest: HumanDatasetManifest) -> HumanDatasetManifest:
    """Return a manifest only after validating its reproducibility metadata."""
    if manifest.access != "author_request" and not manifest.checksum:
        raise ValueError("open or controlled datasets require a SHA-256 checksum")
    if manifest.access == "controlled" and manifest.local_path:
        raise ValueError("controlled data must not point to a repository-local path")
    if manifest.derivation != "downloaded":
        # There is no upstream file to compare against, so the only thing that
        # can be checked mechanically is that the committed reading is stable.
        # That requires a local artefact and a checksum over it.
        if not manifest.local_path:
            raise ValueError("a transcribed or digitised dataset must commit its reading")
        if not manifest.checksum:
            raise ValueError("a transcribed or digitised dataset must checksum its reading")
    return manifest


def validate_manifest_set(manifests):
    """Validate a collection and reject duplicate dataset identifiers."""
    manifests = tuple(validate_manifest(item) for item in manifests)
    ids = [item.dataset_id for item in manifests]
    if len(ids) != len(set(ids)):
        raise ValueError("dataset_id values must be unique")
    return manifests
