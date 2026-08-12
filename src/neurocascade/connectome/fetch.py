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

Why the download is stdlib urllib and not `requests`
----------------------------------------------------

The rest of this codebase depends on numpy and scipy and nothing else at
import time (openpyxl is imported lazily by the one loader that needs it).
A fetch script is not a good reason to add a dependency, and `urlopen`
covers the two things actually needed here: HTTPS with redirects, and a
timeout.
"""

import hashlib
import os
import shutil
import tempfile

from . import sources

#: Default on-disk cache: <repo>/data/connectome. `data/` is gitignored in
#: full; the checksums that make it verifiable live in `sources.py`, which is
#: committed.
DEFAULT_CACHE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "connectome")
)

_CHUNK = 1 << 20  # 1 MiB


class ChecksumMismatch(Exception):
    """Downloaded (or cached) bytes do not match the manifest's SHA-256.

    Deliberately not a subclass of anything catchable by accident. The two
    legitimate responses are: re-download (transfer corruption), or read the
    upstream diff and bump `sources.py` in a commit that explains what moved.
    Editing the expected digest to match whatever arrived is neither.
    """


class MissingCachedFile(Exception):
    """A manifest file is not in the cache, and fetching was not requested."""


def sha256_of(path):
    """Streaming SHA-256 of a file, hex-encoded."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(_CHUNK)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def cache_path(key, cache_dir=None):
    """Where the file for manifest `key` lives (whether or not it exists)."""
    entry = sources.data_file(key)
    root = DEFAULT_CACHE_DIR if cache_dir is None else cache_dir
    return os.path.join(root, entry.filename)


def verify(key, cache_dir=None):
    """Check the cached file against the manifest.

    Returns the path on success. Raises `MissingCachedFile` if it is not
    there, `ChecksumMismatch` if the bytes are wrong. The digest is the
    only thing that decides; the byte count is read purely so the mismatch
    message can say whether the file is a truncated download or a
    different file.
    """
    entry = sources.data_file(key)
    path = cache_path(key, cache_dir)
    if not os.path.exists(path):
        raise MissingCachedFile(
            "%s is not cached at %s; run scripts/fetch_connectome.py" % (key, path)
        )
    actual_size = os.path.getsize(path)
    actual = sha256_of(path)
    if actual != entry.sha256:
        raise ChecksumMismatch(
            "%s: expected sha256 %s (%d bytes, retrieved %s), got %s (%d bytes) "
            "at %s. Manifest version %s. Do NOT edit the expected digest to "
            "match: re-download first, and if it still differs, the upstream "
            "file changed and that change has to be reviewed."
            % (
                key,
                entry.sha256,
                entry.size_bytes,
                entry.retrieved,
                actual,
                actual_size,
                path,
                sources.MANIFEST_VERSION,
            )
        )
    return path


def is_cached(key, cache_dir=None):
    """True if the file is present AND verifies. Never raises for absence."""
    try:
        verify(key, cache_dir)
    except (MissingCachedFile, ChecksumMismatch):
        return False
    return True


def download(key, cache_dir=None, timeout=300, force=False):
    """Download one manifest file into the cache and verify it.

    Downloads to a temporary file in the destination directory and only
    renames into place after the digest matches, so an interrupted or
    corrupt transfer can never leave a file that a later run mistakes for a
    good cache entry. On a digest mismatch the bytes are kept, next to the
    target as `<filename>.rejected`, because the useful next step is almost
    always to look at what actually arrived (an HTML error page, a
    revised ontology) rather than to retry blindly.

    This is the only function in the package that opens a network
    connection.
    """
    from urllib.request import Request, urlopen  # local: keeps import graph clean

    entry = sources.data_file(key)
    path = cache_path(key, cache_dir)
    if not force and is_cached(key, cache_dir):
        return path
    directory = os.path.dirname(path)
    if directory and not os.path.isdir(directory):
        os.makedirs(directory)

    # Springer Nature's CDN serves 403 to the default urllib user agent.
    request = Request(entry.url, headers={"User-Agent": "neurocascade/P3.1"})
    handle, tmp = tempfile.mkstemp(prefix=entry.filename + ".", dir=directory)
    try:
        with os.fdopen(handle, "wb") as out:
            with urlopen(request, timeout=timeout) as response:
                shutil.copyfileobj(response, out, _CHUNK)
        actual = sha256_of(tmp)
        if actual != entry.sha256:
            rejected = path + ".rejected"
            shutil.move(tmp, rejected)
            raise ChecksumMismatch(
                "%s downloaded from %s has sha256 %s, manifest expects %s "
                "(%d bytes, retrieved %s). The bytes received are kept at %s "
                "for inspection. Manifest version %s."
                % (
                    key,
                    entry.url,
                    actual,
                    entry.sha256,
                    entry.size_bytes,
                    entry.retrieved,
                    rejected,
                    sources.MANIFEST_VERSION,
                )
            )
        # mkstemp creates 0600; the cache is ordinary readable data.
        os.chmod(tmp, 0o644)
        shutil.move(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return path


def ensure(key, cache_dir=None, allow_download=True, timeout=300):
    """Return a verified path for `key`, downloading only if allowed.

    `allow_download=False` is what tests and offline runs use: it turns a
    cache miss into `MissingCachedFile` instead of a network call.
    """
    if allow_download:
        return download(key, cache_dir=cache_dir, timeout=timeout)
    return verify(key, cache_dir)


def fetch_all(keys=None, cache_dir=None, timeout=300, force=False, log=None):
    """Download and verify every manifest file (or the named subset).

    Returns {key: path}. `log` is an optional callable taking one string,
    so the CLI can print progress without this module importing anything to
    print with.
    """
    if keys is None:
        keys = [f.key for f in sources.DATA_FILES]
    paths = {}
    for key in keys:
        entry = sources.data_file(key)
        if log is not None:
            log("%s (%d bytes) <- %s" % (key, entry.size_bytes, entry.url))
        paths[key] = download(key, cache_dir=cache_dir, timeout=timeout, force=force)
        if log is not None:
            log("  ok  sha256 %s" % entry.sha256)
    return paths


def checksum_lines():
    """The manifest rendered as `sha256sum -c` input, newest-first order.

    Lets anyone verify a cache directory with the standard tool rather than
    trusting this package's own code to check itself.
    """
    return [
        "%s  %s" % (entry.sha256, entry.filename) for entry in sources.DATA_FILES
    ]
