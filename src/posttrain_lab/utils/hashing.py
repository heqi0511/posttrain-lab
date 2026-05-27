"""Small hashing helpers for immutable experiment artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path


def file_sha256(path):
    """Return the SHA256 hex digest of a file without mutating it."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
