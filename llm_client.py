"""
Shared LLM client for the Campus Assistant demo.

Wraps OpenRouter so every step file can call chat() without
dealing with API setup details.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL = "stepfun/step-3.5-flash:free"

_client: OpenAI | None = None


def get_client() -> OpenAI:
    """Return a cached OpenAI client pointed at OpenRouter."""
    global _client
    if _client is None:
        api_key = os.getenv("OPENROUTER_API_KEY", "")
        if not api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set. "
                "Copy .env.example to .env and add your key from https://openrouter.ai"
            )
        _client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
    return _client


def chat(messages: list[dict]) -> str:
    """
    Send a list of messages to the LLM and return the assistant reply text.

    Parameters
    ----------
    messages : list[dict]
        OpenAI-format message list, e.g.
        [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]

    Returns
    -------
    str
        The assistant's reply text.
    """
    response = get_client().chat.completions.create(
        model=MODEL,
        messages=messages,
    )
    return response.choices[0].message.content or ""
