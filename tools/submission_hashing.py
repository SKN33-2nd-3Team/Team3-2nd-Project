"""Canonical hashing for submission files across Git/Windows EOL settings."""
from __future__ import annotations

import hashlib
from pathlib import Path


TEXT_SUFFIXES = {
    ".css", ".csv", ".html", ".ipynb", ".js", ".json", ".md", ".py",
    ".svg", ".toml", ".tsv", ".txt", ".yaml", ".yml",
}
TEXT_NAMES = {".gitattributes", ".gitignore", "requirements.txt"}


def canonical_bytes(path: Path) -> bytes:
    data = path.read_bytes()
    if path.suffix.lower() in TEXT_SUFFIXES or path.name.lower() in TEXT_NAMES:
        return data.replace(b"\r\n", b"\n")
    return data


def canonical_file_info(path: Path) -> tuple[int, str]:
    data = canonical_bytes(path)
    return len(data), hashlib.sha256(data).hexdigest()
