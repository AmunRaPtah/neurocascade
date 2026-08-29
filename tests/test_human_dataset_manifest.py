import pytest

from neurocascade.human.dataset_manifest import HumanDatasetManifest, validate_manifest, validate_manifest_set


SHA = "a" * 64


def manifest(**kwargs):
    values = dict(
        dataset_id="demo-pet-001",
        layer="pet",
        title="Synthetic PET schema fixture",
        source="https://example.invalid/data",
        citation="Example et al. (2026)",
        license="CC0",
        retrieval_date="2026-08-29",
        checksum=SHA,
        local_path="tests/fixtures/demo.csv",
    )
    values.update(kwargs)
    return HumanDatasetManifest(**values)


def test_open_manifest_requires_checksum_and_local_path_for_redistribution():
    assert validate_manifest(manifest()).redistributable
    with pytest.raises(ValueError, match="checksum"):
        validate_manifest(manifest(checksum=None))


def test_controlled_data_cannot_be_repository_local():
    item = manifest(access="controlled", local_path=None)
    assert not item.redistributable
    with pytest.raises(ValueError, match="repository-local"):
        validate_manifest(manifest(access="controlled"))


def test_author_request_can_remain_unhashed():
    item = manifest(access="author_request", checksum=None, local_path=None)
    assert validate_manifest(item) == item


def test_manifest_set_rejects_duplicates():
    with pytest.raises(ValueError, match="unique"):
        validate_manifest_set([manifest(), manifest()])


def test_date_and_checksum_are_validated():
    with pytest.raises(ValueError, match="ISO"):
        manifest(retrieval_date="yesterday")
    with pytest.raises(ValueError, match="SHA"):
        manifest(checksum="not-a-digest")
