"""Low-memory TF-IDF retriever for deployment environments.

Implements the same public methods as the existing FinancialRetriever without
loading torch or sentence-transformers. Chunks retain their original metadata.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from fincontagion.rag.chunking import chunk_pages


class LightweightFinancialRetriever:
    """Retrieve passages with word and character TF-IDF features."""

    MODEL_NAME = "tfidf-word-char-v1"

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or self.MODEL_NAME
        self.chunks: list[dict[str, Any]] = []
        self.vectorizer: TfidfVectorizer | None = None
        self.matrix = None

    def index_pages(
        self,
        pages: list[dict],
        chunk_size: int = 1000,
        overlap: int = 150,
    ) -> None:
        chunks = chunk_pages(pages, chunk_size=chunk_size, overlap=overlap)
        chunks = [c for c in chunks if str(c.get("text", "")).strip()]
        if not chunks:
            raise ValueError("No text chunks available to index.")

        texts = [str(c["text"]) for c in chunks]
        vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            sublinear_tf=True,
            max_features=100_000,
            dtype="float32",
        )
        try:
            matrix = vectorizer.fit_transform(texts)
        except ValueError as exc:
            raise ValueError("The document contains no searchable text.") from exc

        self.chunks = chunks
        self.vectorizer = vectorizer
        self.matrix = matrix

    
    def save_index(self, index_dir: str | Path) -> None:
        """Persist chunks and vectorizer vocabulary without pickle/joblib."""
        if self.vectorizer is None or self.matrix is None or not self.chunks:
            raise RuntimeError("No indexed documents available to save.")

        path = Path(index_dir)
        path.mkdir(parents=True, exist_ok=True)

        # Convert NumPy scalar values to standard Python values for JSON.
        def json_safe(value):
            if hasattr(value, "item"):
                return value.item()
            raise TypeError(
                f"Object of type {type(value).__name__} is not JSON serializable"
            )

        (path / "chunks.json").write_text(
            json.dumps(self.chunks, ensure_ascii=False, default=json_safe),
            encoding="utf-8",
        )

        metadata = {
            "model_name": self.MODEL_NAME,
            "chunk_count": int(len(self.chunks)),
            "vocabulary": {
                str(term): int(index)
                for term, index in self.vectorizer.vocabulary_.items()
            },
            "idf": self.vectorizer.idf_.tolist(),
            "vectorizer": {
                "ngram_range": [1, 2],
                "sublinear_tf": True,
                "max_features": 100_000,
            },
        }

        (path / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, default=json_safe),
            encoding="utf-8",
        )


    def load_index(self, index_dir: str | Path) -> None:
        path = Path(index_dir)
        metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        if metadata.get("model_name") != self.MODEL_NAME:
            raise ValueError("Saved index is not a lightweight TF-IDF index.")
        chunks = json.loads((path / "chunks.json").read_text(encoding="utf-8"))
        if len(chunks) != metadata.get("chunk_count"):
            raise ValueError("Saved index chunk count does not match metadata.")

        cfg = metadata["vectorizer"]
        vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=tuple(cfg["ngram_range"]),
            sublinear_tf=cfg["sublinear_tf"],
            max_features=cfg["max_features"],
            dtype="float32",
            vocabulary={k: int(v) for k, v in metadata["vocabulary"].items()},
        )
        # Reconstruct the fitted vectorizer from its saved vocabulary and IDF.
        import numpy as np
        from scipy.sparse import csr_matrix

        vectorizer.fixed_vocabulary_ = True
        vectorizer._tfidf.idf_ = np.asarray(metadata["idf"], dtype=np.float32)
        texts = [str(c.get("text", "")) for c in chunks]
        matrix = vectorizer.transform(texts)
        if matrix.shape[0] != len(chunks):
            raise ValueError("Saved TF-IDF matrix does not match chunks.")
        self.chunks, self.vectorizer, self.matrix = chunks, vectorizer, matrix

    def search(self, question: str, top_k: int = 5) -> list[dict]:
        if self.vectorizer is None or self.matrix is None:
            raise RuntimeError("Index documents before searching.")
        if not question.strip():
            raise ValueError("Question cannot be empty.")
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        query = self.vectorizer.transform([question.strip()])
        scores = cosine_similarity(query, self.matrix).ravel()
        # Stable descending sort; omit chunks with no lexical overlap if possible.
        ranked = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))
        selected = [i for i in ranked if scores[i] > 0][:top_k]
        if not selected:
            selected = ranked[: min(top_k, len(ranked))]
        return [{**self.chunks[i], "similarity": float(scores[i])} for i in selected]
