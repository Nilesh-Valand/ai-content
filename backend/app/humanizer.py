"""
LLM-powered content humanization via Groq.

Takes the user's original text and rewrites it to read more naturally —
varied sentence rhythm, concrete detail, less templated phrasing — without
changing its meaning.

User-trained phrase pairs (see db.py) serve as personal writing style guidelines
and few-shot examples for the LLM. Rather than performing rigid exact string substitution,
the LLM intelligently analyzes these phrase pairs to learn the user's tone,
vocabulary preferences, and phrasing style, adapting the content contextually
to match their authentic voice.
"""
import os
from typing import List, Optional

from .db import list_trained_phrases
from .suggestions import DEFAULT_MODEL, get_client

BASE_GUIDELINES = """You are an expert human editor and personalized writing assistant. Rewrite the given text so it reads as naturally human-written while matching the user's personal writing voice and preserving the original meaning, facts, tone, and approximate length.

Guidelines:
- Vary sentence length and structure; avoid repetitive templated patterns.
- Prefer concrete, natural, and expressive phrasing over generic AI filler words.
- Keep the same point of view, register, and intent as the original.
- Do not add new facts, claims, or details that were not implied by the original.
- Do not add commentary, notes, or explanations about the rewrite.
- Do NOT perform rigid, literal string substitution. Apply style preferences flexibly and intelligently according to surrounding context."""

MAX_FEWSHOT_EXAMPLES = 20


def _style_guidelines_block(trained: List[dict]) -> str:
    if not trained:
        return ""
    examples = trained[:MAX_FEWSHOT_EXAMPLES]
    lines = [
        "\n--- USER WRITING STYLE & TRAINED PHRASE GUIDELINES ---",
        "The user has provided the following examples of how they prefer AI-sounding phrases to be rewritten into their personal voice.",
        "Use these examples to infer the user's preferred vocabulary, tone, style, and sentence structure. Adapt the input text dynamically using these style cues:\n",
    ]
    for idx, t in enumerate(examples, start=1):
        lines.append(
            f'Example {idx}:\n'
            f'  AI-sounding phrase: "{t["ai_phrase"].strip()}"\n'
            f'  User\'s preferred style: "{t["humanized_phrase"].strip()}"'
        )
    lines.append("--- END OF STYLE GUIDELINES ---")
    return "\n" + "\n".join(lines)


def _call_groq(system_prompt: str, user_content: str) -> str:
    client = get_client()
    model = os.environ.get("GROQ_MODEL", DEFAULT_MODEL)
    max_tokens = min(4096, max(512, int(len(user_content.split()) * 2.5)))

    completion = client.chat.completions.create(
        model=model,
        temperature=0.7,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    )
    return completion.choices[0].message.content.strip()


def humanize_content(content: str, profile_id: Optional[int] = None) -> str:
    trained = list_trained_phrases(profile_id=profile_id)
    if not trained and profile_id is not None:
        # Fallback to all trained phrases if profile has no phrases defined
        trained = list_trained_phrases()

    system_prompt = (
        BASE_GUIDELINES
        + _style_guidelines_block(trained)
        + '\n\nReturn ONLY the rewritten text, nothing else. No preamble, no markdown headers, no quotation marks wrapping the whole output.'
    )
    return _call_groq(system_prompt, content)

