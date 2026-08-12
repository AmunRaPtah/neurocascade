"""P3.1: download-and-verify for the files in `sources.py`.

The contract this module enforces
---------------------------------

1. **Nothing here touches the network at import time.** Importing
   `neurocascade.connectome` on a machine with no network, or in a test
   run, must not attempt a download and must not fail. Every network call
   lives inside an explicitly called function.

2. **A file is only usable if its SHA-256 matches the manifest.** There is
   no "warn and continue" path. `ChecksumMismatch` is raised, the bad
   bytes are left on disk under a `.rejected` suffix so a human can look at
   them, and nothing downstream sees the file. A connectome that quietly
   changed underneath a published result is worse than a build that stops.

3. **The cache is content, not state.** `data/connectome/` holds exactly
   the bytes published upstream, under the manifest's filenames, and is
   gitignored. Deleting it and re-running the fetch script must reproduce
   it byte for byte. Nothing writes derived or converted files there, so
   there is never a stale intermediate to invalidate.

Where the code actually lives
-----------------------------

All three rules are implemented once, in `neurocascade.datafetch.Fetcher`,
and this module is a thin binding of that machinery to this package's
manifest and cache directory. P4.1 ingests an unrelated body of external
data (Allen ISH expression) under the same contract; sharing the
implementation is what stops the strict half of rule 2 from drifting apart
between the two. The function names and signatures below are unchanged
from the original P3.1 module, so nothing downstream had to move.
"""

import os

from .. import datafetch
from ..datafetch import ChecksumMismatch, MissingCachedFile, sha256_of  # noqa: F401
from . import sources

#: Default on-disk cache: <repo>/data/connectome. `data/` is gitignored in
#: full; the checksums that make it verifiable live in `sources.py`, which is
#: committed.
DEFAULT_CACHE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "connectome")
)

_FETCHER = datafetch.Fetcher(
    manifest=sources,
    default_cache_dir=DEFAULT_CACHE_DIR,
    user_agent="neurocascade/P3.1",
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
