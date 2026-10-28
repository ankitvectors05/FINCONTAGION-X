"""Configurable LLM client: local Ollama or a remote OpenAI-compatible API.

Environment:
  FINCONTAGION_LLM_PROVIDER=ollama|openai_compatible
  OLLAMA_CHAT_URL=http://127.0.0.1:11434/api/chat  (local default)
  OLLAMA_MODEL_NAME=qwen3:1.7b
  LLM_API_BASE_URL=https://provider.example/v1
  LLM_API_KEY=... (set in deployment environment, never commit it)
  LLM_MODEL=provider-model-name
"""
from __future__ import annotations

import os
import requests


def chat(messages: list[dict], *, temperature: float = 0.2, max_tokens: int = 500) -> str:
    provider = os.getenv("FINCONTAGION_LLM_PROVIDER", "ollama").strip().lower()
    timeout = float(os.getenv("LLM_TIMEOUT_SECONDS", "90"))

    if provider == "ollama":
        url = os.getenv("OLLAMA_CHAT_URL", "http://127.0.0.1:11434/api/chat")
        model = os.getenv("OLLAMA_MODEL_NAME", "qwen3:1.7b")
        response = requests.post(
            url,
            json={"model": model, "messages": messages, "stream": False,
                  "think": False, "options": {"temperature": temperature,
                  "num_predict": max_tokens, "num_ctx": 4096}},
            timeout=timeout,
        )
        response.raise_for_status()
        answer = response.json().get("message", {}).get("content", "").strip()
    elif provider == "openai_compatible":
        base_url = os.getenv("LLM_API_BASE_URL", "").rstrip("/")
        api_key = os.getenv("LLM_API_KEY", "")
        model = os.getenv("LLM_MODEL", "")
        if not base_url or not api_key or not model:
            raise RuntimeError(
                "Set LLM_API_BASE_URL, LLM_API_KEY, and LLM_MODEL for the remote provider."
            )
        response = requests.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": model, "messages": messages,
                  "temperature": temperature, "max_tokens": max_tokens},
            timeout=timeout,
        )
        response.raise_for_status()
        choices = response.json().get("choices", [])
        answer = choices[0].get("message", {}).get("content", "").strip() if choices else ""
    else:
        raise RuntimeError("FINCONTAGION_LLM_PROVIDER must be 'ollama' or 'openai_compatible'.")

    if not answer:
        raise RuntimeError("The configured LLM provider returned an empty answer.")
    return answer
