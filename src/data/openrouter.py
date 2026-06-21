"""Thin OpenRouter client: one chat() call with retries + backoff.

Reused for BOTH synthetic generation (Week 1) and judging (Week 3), so it
stays model-agnostic. OpenRouter is OpenAI-compatible; we use raw requests
to keep the mechanics visible (and avoid an extra dependency).
"""
from __future__ import annotations

import os
import time

import requests

from src.utils import get_logger

logger = get_logger("openrouter")
URL = "https://openrouter.ai/api/v1/chat/completions"


class OpenRouterError(RuntimeError):
    pass


def chat(messages: list[dict], model: str = "meta-llama/llama-3.3-70b-instruct",
         temperature: float = 1.0, max_tokens: int = 1024,
         retries: int = 4, timeout: int = 60) -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise OpenRouterError("set OPENROUTER_API_KEY env var")
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages,
               "temperature": temperature, "max_tokens": max_tokens}

    backoff = 2.0
    for attempt in range(1, retries + 1):
        try:
            r = requests.post(URL, headers=headers, json=payload, timeout=timeout)
            if r.status_code == 429 or r.status_code >= 500:  # rate-limit / server -> retry
                raise OpenRouterError(f"retryable {r.status_code}: {r.text[:200]}")
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except (requests.RequestException, OpenRouterError) as e:
            if attempt == retries:
                raise OpenRouterError(f"failed after {retries} attempts: {e}") from e
            logger.warning("attempt %d/%d failed (%s); sleeping %.1fs", attempt, retries, e, backoff)
            time.sleep(backoff)
            backoff *= 2.0