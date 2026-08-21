"""Download-and-verify for the files in `sources.py`.

This is the same contract as `connectome/fetch.py` and
`expression/fetch.py`, bound to the human manifest and the `data/human/`
cache directory. The implementation lives once in
`neurocascade.datafetch.Fetcher`; this module is a thin binding.
"""

import os

from .. import datafetch
from ..datafetch import ChecksumMismatch, MissingCachedFile, sha256_of  # noqa: F401
from . import sources

#: Default on-disk cache: <repo>/data/human. `data/` is gitignored in full;
#: the checksums that make it verifiable live in `sources.py`, which is
#: committed.
DEFAULT_CACHE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "human")
)

_FETCHER = datafetch.Fetcher(
    manifest=sources,
    default_cache_dir=DEFAULT_CACHE_DIR,
    user_agent="neurocascade/human",
)


def cache_path(key, cache_dir=None):
    """Where the file for manifest `key` lives (whether or not it exists)."""
    return _FETCHER.cache_path(key, cache_dir)


def verify(key, cache_dir=None):
    """Check the cached file against the manifest; return the path."""
    return _FETCHER.verify(key, cache_dir)


def is_cached(key, cache_dir=None):
    """True if the file is present AND verifies. Never raises for absence."""
    return _FETCHER.is_cached(key, cache_dir)


def download(key, cache_dir=None, timeout=300, force=False):
    """Download one manifest file into the cache and verify it."""
    return _FETCHER.download(key, cache_dir=cache_dir, timeout=timeout, force=force)


def ensure(key, cache_dir=None, allow_download=True, timeout=300):
    """Return a verified path for `key`, downloading only if allowed."""
    return _FETCHER.ensure(
        key, cache_dir=cache_dir, allow_download=allow_download, timeout=timeout
    )


def fetch_all(keys=None, cache_dir=None, timeout=300, force=False, log=None):
    """Download and verify every manifest file (or the named subset)."""
    return _FETCHER.fetch_all(
        keys=keys, cache_dir=cache_dir, timeout=timeout, force=force, log=log
    )


def checksum_lines():
    """The manifest rendered as `sha256sum -c` input, manifest order."""
    return _FETCHER.checksum_lines()
