"""Evidence-grounded answer generation using the configured LLM provider."""
from __future__ import annotations

from fincontagion.rag.llm_client import chat


def generate_answer(question: str, retrieved_chunks: list[dict]) -> dict:
    sources = []
    evidence = []
    for i, chunk in enumerate(retrieved_chunks, start=1):
        filename = str(chunk.get("filename") or chunk.get("source") or "Unknown source")
        page = chunk.get("page_number", chunk.get("page", "unknown"))
        text = str(chunk.get("text", "")).strip()
        if not text:
            continue
        evidence.append(f"[Source {i}: {filename}, page {page}]\n{text}")
        sources.append({"filename": filename, "page": page, "similarity": chunk.get("similarity")})

    if not evidence:
        return {"answer": "I could not find relevant text in the indexed document to answer this question.", "sources": []}

    system = (
        "You are FINCONTAGION-X, a careful financial research assistant. "
        "Answer only from the supplied evidence. Do not invent facts, figures, "
        "or sources. If evidence is insufficient, say so. Cite claims using "
        "the source filename and page number exactly as supplied. Distinguish "
        "document statements from your interpretation."
    )
    user = f"Question: {question}\n\nRetrieved evidence:\n" + "\n\n".join(evidence)
    answer = chat([{"role": "system", "content": system}, {"role": "user", "content": user}], temperature=0.1, max_tokens=700)
    return {"answer": answer, "sources": sources}
