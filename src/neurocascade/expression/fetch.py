"""P4.1: download-and-verify for the files in `expression/sources.py`.

Same three rules as `connectome/fetch.py`, same implementation
(`neurocascade.datafetch.Fetcher`), a different manifest and a different
cache directory:

1. nothing touches the network at import time, and every loader defaults to
   `allow_download=False`;
2. a file is only usable if its SHA-256 matches the manifest, with no
   "warn and continue" path;
3. `data/expression/` holds exactly the bytes the pinned queries returned,
   is gitignored, and never holds a derived or converted file.

The cache is kept separate from `data/connectome/` so that re-fetching the
ISH data - which is 43 live API queries and the more volatile half of this
project's external data - cannot disturb the connectome cache, and so that
`sha256sum -c` can be run against either directory on its own.
"""

import os

from .. import datafetch
from ..datafetch import ChecksumMismatch, MissingCachedFile, sha256_of  # noqa: F401
from . import sources

#: Default on-disk cache: <repo>/data/expression. `data/` is gitignored in
#: full; the checksums that make it verifiable live in `sources.py`.
DEFAULT_CACHE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "expression")
)

_FETCHER = datafetch.Fetcher(
    manifest=sources,
    default_cache_dir=DEFAULT_CACHE_DIR,
    user_agent="neurocascade/P4.1",
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
