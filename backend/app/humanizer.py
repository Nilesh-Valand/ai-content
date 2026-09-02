"""
LLM-powered content humanization via Groq.

Takes the user's original text and rewrites it to read more naturally —
varied sentence rhythm, concrete detail, less templated phrasing — without
changing its meaning. This is a separate concern from scoring/suggestions:
it doesn't touch the detector's rule-based signals at all.

User-trained phrase pairs (see db.py) are applied two ways:
  1. Exact match: any substring of the content that's an exact (whitespace/
     case/trailing-punctuation-insensitive) match for a trained AI phrase —
     whether it's a whole sentence, a clause, or just a few words — is
     swapped in verbatim for its trained humanized version. Matches are
     protected with sentinel markers so the LLM copies them through
     unchanged instead of rewriting them.
  2. Style guidance: all trained pairs are shown to the LLM as few-shot
     examples so it leans toward the user's preferred tone even for
     phrases it hasn't seen before.
"""
import os
import re
from typing import List, Tuple

from .db import list_trained_phrases
from .suggestions import DEFAULT_MODEL, get_client

BASE_GUIDELINES = """You are an expert human editor. Rewrite the given text so it reads as \
naturally human-written, while preserving its original meaning, facts, tone, and \
approximate length.

Guidelines:
- Vary sentence length and structure; avoid repetitive templated patterns.
- Prefer concrete, specific phrasing over generic filler.
- Keep the same point of view, register (formal/casual), and intent as the original.
- Do not add new facts, claims, or details that were not implied by the original.
- Do not add commentary, notes, or explanations about the rewrite."""

MAX_FEWSHOT_EXAMPLES = 12
LOCK_OPEN = "%%%LOCK%%%"
LOCK_CLOSE = "%%%ENDLOCK%%%"


def _normalize(text: str) -> str:
    collapsed = " ".join(text.strip().split())
    return collapsed.rstrip(" .!?,;:").lower()


def _phrase_regex_part(phrase: str) -> str:
    tokens = phrase.strip().split()
    return r"\s+".join(re.escape(t) for t in tokens)


def _find_exact_matches(content: str, trained: List[dict]) -> List[Tuple[int, int, str]]:
    """Locate non-overlapping occurrences of trained AI phrases anywhere in the
    content — not just whole sentences. Longer phrases win when matches overlap."""
    ordered = sorted(trained, key=lambda t: len(t["ai_phrase"]), reverse=True)
    lookup = {_normalize(t["ai_phrase"]): t["humanized_phrase"].strip() for t in ordered}
    pattern = re.compile(
        "(?:" + "|".join(_phrase_regex_part(t["ai_phrase"]) for t in ordered) + ")",
        re.IGNORECASE,
    )

    matches: List[Tuple[int, int, str]] = []
    occupied_end = -1
    for m in pattern.finditer(content):
        if m.start() < occupied_end:
            continue
        replacement = lookup.get(_normalize(m.group(0)))
        if replacement is None:
            continue
        matches.append((m.start(), m.end(), replacement))
        occupied_end = m.end()
    return matches


def _fewshot_block(trained: List[dict]) -> str:
    if not trained:
        return ""
    examples = trained[:MAX_FEWSHOT_EXAMPLES]
    lines = [
        "\nHere are example pairs showing the exact tone and phrasing style this "
        "user prefers. Lean toward this style wherever it applies:"
    ]
    for t in examples:
        lines.append(
            f'- AI-sounding: "{t["ai_phrase"].strip()}"\n'
            f'  Preferred human rewrite: "{t["humanized_phrase"].strip()}"'
        )
    return "\n".join(lines)


def _call_groq(system_prompt: str, user_content: str) -> str:
    client = get_client()
    model = os.environ.get("GROQ_MODEL", DEFAULT_MODEL)
    max_tokens = min(4096, max(512, int(len(user_content.split()) * 2.5)))

    completion = client.chat.completions.create(
        model=model,
        temperature=0.7,
        max_tokens=max_tokens,
        reasoning_effort="low",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    )
    return completion.choices[0].message.content.strip()


def _humanize_whole_text(content: str, trained: List[dict]) -> str:
    system_prompt = (
        BASE_GUIDELINES
        + _fewshot_block(trained)
        + '\n\nReturn ONLY the rewritten text, nothing else. No preamble, no '
          "markdown headers, no quotation marks wrapping the whole output."
    )
    return _call_groq(system_prompt, content)


def _humanize_locked_text(locked_content: str, trained: List[dict]) -> str:
    system_prompt = (
        BASE_GUIDELINES
        + _fewshot_block(trained)
        + f"\n\nCRITICAL RULE — HIGHEST PRIORITY: The text contains segments wrapped "
          f"like this: {LOCK_OPEN}some text{LOCK_CLOSE}. You MUST copy the text inside "
          f"every {LOCK_OPEN}...{LOCK_CLOSE} pair character-for-character, with zero "
          f"changes — same words, same punctuation, same capitalization. Do NOT "
          f"paraphrase, smooth, merge, or adjust it in any way, even if it makes the "
          f"surrounding sentence read slightly awkwardly. Remove only the {LOCK_OPEN} "
          f"and {LOCK_CLOSE} marker tokens themselves from your output; keep the text "
          f"between them untouched. This rule overrides every other guideline above "
          f"when they conflict. Rewrite ONLY the text outside the markers, following "
          f"the guidelines above."
          "\n\nReturn ONLY the rewritten text, nothing else. No preamble, no "
          "markdown headers, no quotation marks wrapping the whole output."
    )
    rewritten = _call_groq(system_prompt, locked_content)
    # Safety net in case the model echoes a marker back despite instructions.
    return rewritten.replace(LOCK_OPEN, "").replace(LOCK_CLOSE, "")


def humanize_content(content: str) -> str:
    trained = list_trained_phrases()

    if not trained:
        return _humanize_whole_text(content, trained)

    matches = _find_exact_matches(content, trained)
    if not matches:
        return _humanize_whole_text(content, trained)

    # Fast path: the entire content is itself exactly one trained phrase.
    if len(matches) == 1 and matches[0][0] == 0 and matches[0][1] >= len(content.rstrip()):
        return matches[0][2]

    pieces = []
    cursor = 0
    for start, end, replacement in matches:
        pieces.append(content[cursor:start])
        pieces.append(f"{LOCK_OPEN}{replacement}{LOCK_CLOSE}")
        cursor = end
    pieces.append(content[cursor:])
    locked_content = "".join(pieces)

    return _humanize_locked_text(locked_content, trained)
