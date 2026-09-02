"""
LLM-powered content humanization via Groq.

Takes the user's original text and rewrites it to read more naturally —
varied sentence rhythm, concrete detail, less templated phrasing — without
changing its meaning. This is a separate concern from scoring/suggestions:
it doesn't touch the detector's rule-based signals at all.
"""
import os

from .suggestions import DEFAULT_MODEL, get_client

HUMANIZE_SYSTEM_PROMPT = """You are an expert human editor. Rewrite the given \
text so it reads as naturally human-written, while preserving its original \
meaning, facts, tone, and approximate length.

Guidelines:
- Vary sentence length and structure; avoid repetitive templated patterns.
- Prefer concrete, specific phrasing over generic filler.
- Keep the same point of view, register (formal/casual), and intent as the original.
- Do not add new facts, claims, or details that were not implied by the original.
- Do not add commentary, notes, or explanations about the rewrite.

Return ONLY the rewritten text, nothing else. No preamble, no markdown \
headers, no quotation marks wrapping the whole output."""


def humanize_content(content: str) -> str:
    client = get_client()
    model = os.environ.get("GROQ_MODEL", DEFAULT_MODEL)

    max_tokens = min(4096, max(512, int(len(content.split()) * 2.5)))

    completion = client.chat.completions.create(
        model=model,
        temperature=0.7,
        max_tokens=max_tokens,
        reasoning_effort="low",
        messages=[
            {"role": "system", "content": HUMANIZE_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
    )
    return completion.choices[0].message.content.strip()
