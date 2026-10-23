
import os
import requests


OLLAMA_URL = os.getenv(
    "OLLAMA_GENERATE_URL",
    "http://localhost:11434/api/generate",
)
MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "qwen3:1.7b")

def generate_answer(question: str, retrieved_chunks: list[dict]) -> dict:
    """
    Generate an evidence-grounded answer from retrieved document chunks.
    Each chunk should contain: source, page, text.
    """
    if not question.strip():
        raise ValueError("Question cannot be empty.")

    if not retrieved_chunks:
        return {
            "answer": "I could not find relevant evidence in the supplied documents.",
            "sources": [],
        }

    context_parts = []
    sources = []
    seen_sources = set()

    for i, chunk in enumerate(retrieved_chunks, start=1):
        source = chunk.get("source", "Unknown document")
        page = chunk.get("page", "Unknown")
        text = chunk.get("text", "").strip()

        if not text:
            continue

        context_parts.append(
            f"[Source {i}: {source}, page {page}]\n{text}"
        )

        source_key = (source, page)
        if source_key not in seen_sources:
            sources.append({"source": source, "page": page})
            seen_sources.add(source_key)

    if not context_parts:
        return {
            "answer": "The retrieved passages contain no usable text.",
            "sources": [],
        }

    context = "\n\n".join(context_parts)

    prompt = f"""
You are the financial research assistant for FINCONTAGION-X.

Answer the user's question using ONLY the document evidence below.

Rules:
- Answer using only the document evidence provided.
- Do not invent facts, statistics, or conclusions.
- If evidence is insufficient, say so clearly.
- Cite evidence using the exact filename and page number shown in each passage header.
- Never use labels such as [Source 1] or [Source 2] in the final answer.
- Do not cite a page unless the supplied passage supports the claim.
- Keep the answer concise and easy to understand.

DOCUMENT EVIDENCE:
{context}

USER QUESTION:
{question}

EVIDENCE-BASED ANSWER:
""".strip()

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "options": {
                "num_predict": 250,
                "num_ctx": 4096,
                "temperature": 0.1,
            },
        },
        timeout=240,
    )
    response.raise_for_status()

    answer = response.json().get("response", "").strip()

    return {
        "answer": answer or "The model returned an empty answer.",
        "sources": sources,
    }
