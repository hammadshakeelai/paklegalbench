"""
Generation layer. Free tiers only, with fallback.

Order: Groq (fast, 30 rpm / 1000 rpd) -> Gemini (big context) -> OpenRouter.
Whichever has a key set gets tried first, in that order.

Model IDs move around on free tiers. Override with env vars rather than
editing this file.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

TIMEOUT = 45

SYSTEM_PROMPT = """You are a legal research assistant for Pakistani law. You answer \
ONLY from the provisions supplied to you in CONTEXT.

Rules, in order of importance:

1. Every legal assertion you make must be followed by its citation in square \
brackets, e.g. [PPC Section 420] or [Constitution, Article 199]. If you cannot \
attach a citation from CONTEXT to a statement, do not make that statement.

2. If CONTEXT does not contain the answer, say so plainly and stop. Do not fill \
the gap from your own knowledge of Pakistani law. A short "the provisions I \
retrieved do not cover this" is a correct and useful answer.

3. If the question assumes a provision that does not appear in CONTEXT (for \
example a section number that does not exist), say that you could not find that \
provision rather than describing what it might contain.

4. Never state or imply that a provision is currently in force, unamended, or \
good law. You only have the text, not its amendment history.

5. Be brief. Two to five sentences for most questions. No preamble, no \
restating the question.

6. Close with one line: "Verify against the primary source before relying on \
this." """


class NoProviderError(RuntimeError):
    pass


def _post(url: str, payload: dict, headers: dict) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers=headers, method="POST"
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return json.loads(r.read().decode())


def _groq(messages: list[dict]) -> str:
    key = os.environ["GROQ_API_KEY"]
    model = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
    data = _post(
        "https://api.groq.com/openai/v1/chat/completions",
        {"model": model, "messages": messages, "temperature": 0.1, "max_tokens": 700},
        {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    return data["choices"][0]["message"]["content"]


def _gemini(messages: list[dict]) -> str:
    key = os.environ["GEMINI_API_KEY"]
    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    system = "\n".join(m["content"] for m in messages if m["role"] == "system")
    turns = [
        {"role": "model" if m["role"] == "assistant" else "user",
         "parts": [{"text": m["content"]}]}
        for m in messages if m["role"] != "system"
    ]
    data = _post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}",
        {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": turns,
            "generationConfig": {"temperature": 0.1, "maxOutputTokens": 700},
        },
        {"Content-Type": "application/json"},
    )
    return data["candidates"][0]["content"]["parts"][0]["text"]


def _openrouter(messages: list[dict]) -> str:
    key = os.environ["OPENROUTER_API_KEY"]
    model = os.environ.get("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free")
    data = _post(
        "https://openrouter.ai/api/v1/chat/completions",
        {"model": model, "messages": messages, "temperature": 0.1, "max_tokens": 700},
        {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    return data["choices"][0]["message"]["content"]


PROVIDERS = [
    ("groq", "GROQ_API_KEY", _groq),
    ("gemini", "GEMINI_API_KEY", _gemini),
    ("openrouter", "OPENROUTER_API_KEY", _openrouter),
]


def available_providers() -> list[str]:
    return [name for name, env, _ in PROVIDERS if os.environ.get(env)]


def build_context(hits) -> str:
    """hits: list[index.Hit] -> the CONTEXT block."""
    parts = []
    for h in hits:
        c = h.chunk
        parts.append(
            f"--- {c.citation()} | {c.marginal_note} | {c.act} ---\n{c.text}"
        )
    return "\n\n".join(parts)


def answer(question: str, hits, history: list[dict] | None = None) -> tuple[str, str]:
    """Returns (answer_text, provider_used). Raises NoProviderError if no key set."""
    context = build_context(hits)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in (history or [])[-4:]:
        messages.append(turn)
    messages.append(
        {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {question}"}
    )

    errors = []
    for name, env, fn in PROVIDERS:
        if not os.environ.get(env):
            continue
        try:
            return fn(messages), name
        except (urllib.error.HTTPError, urllib.error.URLError, KeyError, TimeoutError) as e:
            errors.append(f"{name}: {e}")
            continue

    if not available_providers():
        raise NoProviderError(
            "No API key found. Set GROQ_API_KEY, GEMINI_API_KEY or OPENROUTER_API_KEY."
        )
    raise NoProviderError("All providers failed. " + " | ".join(errors))
