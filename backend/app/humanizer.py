"""
LLM-powered content humanization via Groq.

Takes the user's original text and rewrites it to read more naturally —
varied sentence rhythm, concrete detail, less templated phrasing — without
changing its meaning. This is a separate concern from scoring/suggestions:
it doesn't touch the detector's rule-based signals at all.

User-trained phrase pairs (see db.py) are applied two ways:
  1. Exact match: a sentence that's an exact (whitespace/case/trailing-
     punctuation-insensitive) match for a trained AI phrase is swapped
     in verbatim for its trained humanized version — never touched by the LLM.
  2. Style guidance: all trained pairs are shown to the LLM as few-shot
     examples so it leans toward the user's preferred tone even for
     sentences it hasn't seen before.
"""
import json
import os
from typing import List

from .analyzer import get_nlp
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


def _normalize(text: str) -> str:
    collapsed = " ".join(text.strip().split())
    return collapsed.rstrip(" .!?").lower()


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


def _humanize_whole_text(content: str, trained: List[dict]) -> str:
    client = get_client()
    model = os.environ.get("GROQ_MODEL", DEFAULT_MODEL)

    system_prompt = (
        BASE_GUIDELINES
        + _fewshot_block(trained)
        + '\n\nReturn ONLY the rewritten text, nothing else. No preamble, no '
          "markdown headers, no quotation marks wrapping the whole output."
    )
    max_tokens = min(4096, max(512, int(len(content.split()) * 2.5)))

    completion = client.chat.completions.create(
        model=model,
        temperature=0.7,
        max_tokens=max_tokens,
        reasoning_effort="low",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ],
    )
    return completion.choices[0].message.content.strip()


def _humanize_chunks(chunks: List[str], trained: List[dict]) -> List[str]:
    client = get_client()
    model = os.environ.get("GROQ_MODEL", DEFAULT_MODEL)

    system_prompt = (
        BASE_GUIDELINES
        + _fewshot_block(trained)
        + '\n\nYou will receive a JSON array of text chunks under "chunks". Rewrite '
          "EACH chunk independently following the same guidelines. Return ONLY a "
          "JSON array of the rewritten chunk strings, in the same order and the "
          "same length as the input array. No markdown, no preamble, no code fences."
    )
    max_tokens = min(4096, max(512, int(sum(len(c.split()) for c in chunks) * 3.0)))

    completion = client.chat.completions.create(
        model=model,
        temperature=0.7,
        max_tokens=max_tokens,
        reasoning_effort="low",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps({"chunks": chunks}, ensure_ascii=False)},
        ],
    )
    raw = completion.choices[0].message.content.strip()
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    parsed = json.loads(raw)

    if not isinstance(parsed, list) or len(parsed) != len(chunks):
        raise ValueError("LLM chunk rewrite response did not match expected shape")

    return [str(item).strip() for item in parsed]


def humanize_content(content: str) -> str:
    trained = list_trained_phrases()

    if not trained:
        return _humanize_whole_text(content, trained)

    nlp = get_nlp()
    sentences = [s.text.strip() for s in nlp(content).sents if s.text.strip()]
    if not sentences:
        return _humanize_whole_text(content, trained)

    lookup = {_normalize(t["ai_phrase"]): t["humanized_phrase"].strip() for t in trained}

    matched = [_normalize(s) in lookup for s in sentences]
    if not any(matched):
        return _humanize_whole_text(content, trained)

    final = [lookup[_normalize(s)] if m else None for s, m in zip(sentences, matched)]

    if all(matched):
        return " ".join(final)

    # Group consecutive unmatched sentences into chunks so nearby sentences
    # keep their local context when sent to the LLM together.
    chunk_spans = []
    i = 0
    while i < len(sentences):
        if matched[i]:
            i += 1
            continue
        j = i
        while j < len(sentences) and not matched[j]:
            j += 1
        chunk_spans.append((i, j))
        i = j

    rewritten_chunks = _humanize_chunks(
        [" ".join(sentences[start:end]) for start, end in chunk_spans], trained
    )

    for (start, end), rewritten in zip(chunk_spans, rewritten_chunks):
        final[start] = rewritten
        for k in range(start + 1, end):
            final[k] = None

    return " ".join(part for part in final if part)
