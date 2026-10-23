
from typing import Any


def chunk_pages(
    pages: list[dict[str, Any]],
    chunk_size: int = 1000,
    overlap: int = 150,
) -> list[dict[str, Any]]:
    """Split extracted page text into overlapping chunks."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size.")

    chunks = []
    chunk_id = 0

    for page in pages:
        text = page["text"].strip()

        if not text:
            continue

        start = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))
            content = text[start:end].strip()

            if content:
                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "source": page["source"],
                        "page": page["page"],
                        "text": content,
                    }
                )
                chunk_id += 1

            if end == len(text):
                break

            start = end - overlap

    return chunks
