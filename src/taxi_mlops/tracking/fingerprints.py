"""Stable content fingerprints for data, configuration, and artifact lineage."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of one file without loading it all into memory."""

    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_files(paths: Iterable[str | Path], *, root: str | Path) -> str:
    """Hash file names and content into one order-independent lineage fingerprint."""

    root = Path(root).resolve()
    normalized: list[tuple[str, Path]] = []
    for value in paths:
        path = Path(value).resolve()
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError as error:
            raise ValueError(f"lineage file is outside root: {path}") from error
        if not path.is_file():
            raise FileNotFoundError(path)
        normalized.append((relative, path))

    digest = hashlib.sha256()
    for relative, path in sorted(normalized):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()
