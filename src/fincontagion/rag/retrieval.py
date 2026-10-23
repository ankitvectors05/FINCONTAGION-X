import json
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer, util

from fincontagion.rag.chunking import chunk_pages
from fincontagion.rag.ingestion import extract_pdf_pages


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class FinancialRetriever:
    """Retrieve relevant financial-document passages using embeddings."""

    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.chunks = []
        self.embeddings = None

    def index_pages(self, pages: list[dict], chunk_size: int = 1000, overlap: int = 150):
        """Chunk extracted pages and encode them for semantic search."""

        self.chunks = chunk_pages(
            pages,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        if not self.chunks:
            raise ValueError("No text chunks available to index.")

        texts = [chunk["text"] for chunk in self.chunks]

        self.embeddings = self.model.encode(
            texts,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        
    def save_index(self, index_dir: str | Path) -> None:
        """Save document chunks and embeddings to disk."""

        if self.embeddings is None or not self.chunks:
            raise RuntimeError("No indexed documents available to save.")

        index_path = Path(index_dir)
        index_path.mkdir(parents=True, exist_ok=True)

        with (index_path / "chunks.json").open("w", encoding="utf-8") as file:
            json.dump(self.chunks, file, ensure_ascii=False, indent=2)

        torch.save(
            self.embeddings.cpu(),
            index_path / "embeddings.pt",
        )

        metadata = {
            "model_name": MODEL_NAME,
            "chunk_count": len(self.chunks),
        }

        with (index_path / "metadata.json").open("w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2)

    def load_index(self, index_dir: str | Path) -> None:
        """Load previously saved document chunks and embeddings."""

        index_path = Path(index_dir)

        with (index_path / "metadata.json").open("r", encoding="utf-8") as file:
            metadata = json.load(file)

        if metadata["model_name"] != self.model_name:
            raise ValueError("Saved index uses a different embedding model.")

        with (index_path / "chunks.json").open("r", encoding="utf-8") as file:
            chunks = json.load(file)

        embeddings = torch.load(
            index_path / "embeddings.pt",
            map_location="cpu",
            weights_only=True,
        )

        if len(chunks) != metadata["chunk_count"]:
            raise ValueError("Saved index chunk count does not match metadata.")

        if embeddings.shape[0] != len(chunks):
            raise ValueError("Saved embeddings do not match the saved chunks.")

        self.chunks = chunks
        self.embeddings = embeddings


    def search(self, question: str, top_k: int = 5) -> list[dict]:
        """Return the most semantically relevant chunks."""

        if self.embeddings is None:
            raise RuntimeError("Index documents before searching.")

        if not question.strip():
            raise ValueError("Question cannot be empty.")

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        query_embedding = self.model.encode(
            question,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )

        scores = util.cos_sim(query_embedding, self.embeddings)[0]
        k = min(top_k, len(self.chunks))
        top_results = scores.topk(k=k)

        results = []

        for score, index in zip(
            top_results.values.tolist(),
            top_results.indices.tolist(),
        ):
            chunk = self.chunks[index]

            results.append(
                {
                    **chunk,
                    "similarity": float(score),
                }
            )

        return results


if __name__ == "__main__":
    pdf_name = "rbi_fsr_june_2025.pdf"
    pages = extract_pdf_pages(pdf_name)

    retriever = FinancialRetriever()
    retriever.index_pages(pages)

    question = "What risks could threaten the stability of India's financial system?"
    results = retriever.search(question, top_k=5)

    print(f"\nQuestion: {question}")
    print(f"Indexed chunks: {len(retriever.chunks)}")

    for rank, result in enumerate(results, start=1):
        print(f"\n--- Result {rank} ---")
        print(f"Source: {result['source']}, page: {result['page']}")
        print(f"Similarity: {result['similarity']:.4f}")
        print(result["text"][:500])
