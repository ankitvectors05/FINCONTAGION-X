
import hashlib
from pathlib import Path


def calculate_sha256(file_path: str | Path) -> str:
    """Return the SHA-256 hash of a file without loading it all into memory."""

    file_path = Path(file_path)
    sha256 = hashlib.sha256()

    with file_path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def get_document_directory(
    file_hash: str,
    storage_root: str | Path = "data/processed/user_documents",
) -> Path:
    """Return the storage directory for a document identified by its hash."""

    if len(file_hash) != 64 or any(
        character not in "0123456789abcdef" for character in file_hash
    ):
        raise ValueError("Expected a lowercase SHA-256 hash.")

    return Path(storage_root) / file_hash
