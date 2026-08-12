"""Download-and-verify against a checksummed manifest, shared by every
external-data package in this project.

Why this module exists
----------------------

P3.1 established the contract for ingesting published data: a committed
manifest of URLs and SHA-256 digests, a gitignored cache holding exactly
the published bytes, and a hard failure (never a warning) when the two
disagree. P4.1 ingests a second, unrelated body of external data (Allen
ISH expression) and needs exactly the same contract.

The alternative to this module was copying `connectome/fetch.py` and
editing two names, which would have left two implementations of "what
happens on a checksum mismatch" free to drift apart. The behaviour that
must not drift is precisely the strict part: reject, keep the bytes as
`.rejected`, and say in the error message that editing the expected digest
is not one of the legitimate responses.

So the logic lives here once, parameterised by

  * a **manifest module**, which must expose `MANIFEST_VERSION`,
    `DATA_FILES` and `data_file(key)` - `connectome/sources.py` and
    `expression/sources.py` both do; and
  * a **default cache directory**, because the two bodies of data are kept
    apart on disk (`data/connectome/`, `data/expression/`) so that one can
    be re-fetched without touching the other.

`connectome.fetch` and `expression.fetch` are thin modules over a single
`Fetcher` instance each, and keep their module-level function API so that
nothing downstream of P3.1 changed.

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

_CHUNK = 1 << 20  # 1 MiB


class DataFile(object):
    """One external file: where it comes from, what it hashes to, what it means.

    `sha256` and `size_bytes` are measured values, recorded from the bytes
    actually retrieved on `retrieved`. They are not to be edited to make a
    failing fetch pass; a mismatch means the upstream file changed and the
    change has to be looked at (see `ChecksumMismatch`).

    `volatility` is either "static" (an archive server that is expected to
    serve the same bytes forever) or "live-api" (a URL the publisher renders
    from a database per request, so the bytes can move without notice). Both
    are checksummed; the label exists so the failure can be read correctly
    when it happens.
    """

    def __init__(
        self,
        key,
        filename,
        url,
        sha256,
        size_bytes,
        citation,
        description,
        units,
        retrieved,
        volatility="static",
    ):
        self.key = key
        self.filename = filename
        self.url = url
        self.sha256 = sha256
        self.size_bytes = size_bytes
        self.citation = citation
        self.description = description
        self.units = units
        self.retrieved = retrieved
        self.volatility = volatility

    def __repr__(self):
        return "%s(%r, %d bytes, sha256=%s...)" % (
            type(self).__name__,
            self.key,
            self.size_bytes,
            self.sha256[:12],
        )


class ChecksumMismatch(Exception):
    """Downloaded (or cached) bytes do not match the manifest's SHA-256.

    Deliberately not a subclass of anything catchable by accident. The two
    legitimate responses are: re-download (transfer corruption), or read the
    upstream diff and bump the manifest in a commit that explains what moved.
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


class Fetcher(object):
    """Verified download/cache for one manifest module.

    `manifest` is a module exposing `MANIFEST_VERSION`, `DATA_FILES` and
    `data_file(key)`. `user_agent` is sent on every request because at least
    one publisher's CDN (Springer Nature's, for the Oh et al. 2014
    supplementary table) serves 403 to urllib's default.
    """

    def __init__(self, manifest, default_cache_dir, user_agent):
        self.manifest = manifest
        self.default_cache_dir = default_cache_dir
        self.user_agent = user_agent

    # --- paths and verification -------------------------------------------

    def cache_path(self, key, cache_dir=None):
        """Where the file for manifest `key` lives (whether or not it exists)."""
        entry = self.manifest.data_file(key)
        root = self.default_cache_dir if cache_dir is None else cache_dir
        return os.path.join(root, entry.filename)

    def verify(self, key, cache_dir=None):
        """Check the cached file against the manifest.

        Returns the path on success. Raises `MissingCachedFile` if it is not
        there, `ChecksumMismatch` if the bytes are wrong. The digest is the
        only thing that decides; the byte count is read purely so the
        mismatch message can say whether the file is a truncated download or
        a different file.
        """
        entry = self.manifest.data_file(key)
        path = self.cache_path(key, cache_dir)
        if not os.path.exists(path):
            raise MissingCachedFile(
                "%s is not cached at %s; run %s" % (key, path, self.fetch_hint())
            )
        actual_size = os.path.getsize(path)
        actual = sha256_of(path)
        if actual != entry.sha256:
            raise ChecksumMismatch(
                "%s: expected sha256 %s (%d bytes, retrieved %s), got %s "
                "(%d bytes) at %s. Manifest version %s. Do NOT edit the "
                "expected digest to match: re-download first, and if it still "
                "differs, the upstream file changed and that change has to be "
                "reviewed."
                % (
                    key,
                    entry.sha256,
                    entry.size_bytes,
                    entry.retrieved,
                    actual,
                    actual_size,
                    path,
                    self.manifest.MANIFEST_VERSION,
                )
            )
        return path

    def fetch_hint(self):
        """The command a human should run; overridden per manifest."""
        return getattr(self.manifest, "FETCH_COMMAND", "the fetch script")

    def is_cached(self, key, cache_dir=None):
        """True if the file is present AND verifies. Never raises for absence."""
        try:
            self.verify(key, cache_dir)
        except (MissingCachedFile, ChecksumMismatch):
            return False
        return True

    # --- network -----------------------------------------------------------

    def download(self, key, cache_dir=None, timeout=300, force=False):
        """Download one manifest file into the cache and verify it.

        Downloads to a temporary file in the destination directory and only
        renames into place after the digest matches, so an interrupted or
        corrupt transfer can never leave a file that a later run mistakes
        for a good cache entry. On a digest mismatch the bytes are kept,
        next to the target as `<filename>.rejected`, because the useful next
        step is almost always to look at what actually arrived (an HTML
        error page, a revised ontology) rather than to retry blindly.

        This is the only function in the project that opens a network
        connection.
        """
        from urllib.request import Request, urlopen  # local: import graph

        entry = self.manifest.data_file(key)
        path = self.cache_path(key, cache_dir)
        if not force and self.is_cached(key, cache_dir):
            return path
        directory = os.path.dirname(path)
        if directory and not os.path.isdir(directory):
            os.makedirs(directory)

        request = Request(entry.url, headers={"User-Agent": self.user_agent})
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
                    "(%d bytes, retrieved %s). The bytes received are kept at "
                    "%s for inspection. Manifest version %s."
                    % (
                        key,
                        entry.url,
                        actual,
                        entry.sha256,
                        entry.size_bytes,
                        entry.retrieved,
                        rejected,
                        self.manifest.MANIFEST_VERSION,
                    )
                )
            # mkstemp creates 0600; the cache is ordinary readable data.
            os.chmod(tmp, 0o644)
            shutil.move(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
        return path

    def ensure(self, key, cache_dir=None, allow_download=True, timeout=300):
        """Return a verified path for `key`, downloading only if allowed.

        `allow_download=False` is what tests and offline runs use: it turns a
        cache miss into `MissingCachedFile` instead of a network call.
        """
        if allow_download:
            return self.download(key, cache_dir=cache_dir, timeout=timeout)
        return self.verify(key, cache_dir)

    def fetch_all(self, keys=None, cache_dir=None, timeout=300, force=False, log=None):
        """Download and verify every manifest file (or the named subset).

        Returns {key: path}. `log` is an optional callable taking one string,
        so a CLI can print progress without this module importing anything to
        print with.
        """
        if keys is None:
            keys = [f.key for f in self.manifest.DATA_FILES]
        paths = {}
        for key in keys:
            entry = self.manifest.data_file(key)
            if log is not None:
                log("%s (%d bytes) <- %s" % (key, entry.size_bytes, entry.url))
            paths[key] = self.download(
                key, cache_dir=cache_dir, timeout=timeout, force=force
            )
            if log is not None:
                log("  ok  sha256 %s" % entry.sha256)
        return paths

    # --- reporting ---------------------------------------------------------

    def checksum_lines(self):
        """The manifest rendered as `sha256sum -c` input, manifest order.

        Lets anyone verify a cache directory with the standard tool rather
        than trusting this package's own code to check itself.
        """
        return [
            "%s  %s" % (entry.sha256, entry.filename)
            for entry in self.manifest.DATA_FILES
        ]
