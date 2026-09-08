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

Three things steer that beyond the raw content itself:
  1. The selected profile's own name/description (its "identity"), so the
     model actually knows what style it's writing in — not just which DB
     rows happen to be tagged with that profile.
  2. A relevance-ranked subset of that profile's trained phrases, so the
     few-shot examples shown to the model are the ones most likely to
     actually apply to the text being rewritten right now.
  3. A second, lower-temperature quality-check pass over the rewrite that
     catches and minimally repairs genuine defects (same-root repetition,
     grammar errors, meaning drift, profile-contradicting tone) without
     rewriting wholesale — and fails safe to the original rewrite if
     anything about that pass goes wrong.
  4. A final deterministic rule-based scrub (rule_scrubber.py) that removes
     whichever of the detector's own flaggable surface patterns survived —
     AI vocabulary, sentence-opener crutches, em dash overuse — since this
     app owns the detector too and can target its exact signals directly
     instead of hoping the LLM avoided them all.
"""
import os
import re
from typing import List, Optional

from .db import get_profile, list_trained_phrases
from .rule_scrubber import scrub_ai_signals
from .suggestions import DEFAULT_MODEL, get_client

BASE_GUIDELINES = """You are an expert human editor and personalized writing assistant. Rewrite the given text so it reads as naturally human-written while matching the user's personal writing voice and preserving the original meaning, facts, tone, and approximate length.

Guidelines:
- Vary sentence length and structure; avoid repetitive templated patterns.
- Prefer concrete, natural, and expressive phrasing over generic AI filler words.
- Keep the same point of view, register, and intent as the original.
- Do not add new facts, claims, or details that were not implied by the original.
- Do not add commentary, notes, or explanations about the rewrite.
- Do NOT perform rigid, literal string substitution. Apply style preferences flexibly and intelligently according to surrounding context."""

MIN_FEWSHOT_EXAMPLES = 3
MAX_FEWSHOT_EXAMPLES = 8

QUALITY_CHECK_SYSTEM_PROMPT = """You are a meticulous copy editor doing a final quality pass on an already-rewritten piece of text. You will be shown the ORIGINAL text and a REWRITE of it. Your only job is to catch and fix genuine defects in the REWRITE:
- Same-root word repetition or awkward wordplay (e.g. "spotting the spots", "using the use of")
- Grammar or punctuation errors
- Meaning drift — the rewrite says something different from the original
- Tone that contradicts the target writing profile, if one is given below

Rules:
- Make the SMALLEST possible edit to fix a real problem. Do not rewrite wholesale.
- If the REWRITE has no genuine defects, return it completely unchanged, word for word.
- Never change the meaning, add new information, or significantly change the length.
- Return ONLY the final text (edited or unchanged), nothing else — no preamble, no explanation, no markdown, no quotation marks wrapping the output."""

# Common function words excluded from relevance scoring so overlap reflects
# shared topic/content words rather than shared grammar.
_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "and", "or", "but", "if", "of", "in", "on", "at", "to", "for", "with",
    "as", "by", "that", "this", "these", "those", "it", "its", "from",
    "we", "they", "he", "she", "you", "i", "not", "can", "will", "would",
    "should", "could", "do", "does", "did", "has", "have", "had", "their",
})


def _profile_identity_block(profile: Optional[dict]) -> str:
    """Surface the profile's own name/description to the model, so a style
    like "LinkedIn" is an actual instruction, not just a hidden DB filter."""
    if not profile:
        return ""

    description = (profile.get("description") or "").strip()
    lines = [
        "\n--- TARGET WRITING PROFILE ---",
        f'Rewrite this content to match the "{profile["name"]}" writing style.',
    ]
    if description:
        lines.append(f"Style description: {description}")
    lines.append("--- END OF TARGET WRITING PROFILE ---")
    return "\n" + "\n".join(lines)


def _profile_context_line(profile: Optional[dict]) -> str:
    """A short profile mention for the quality-check prompt — describes the
    target style for context, without the rewrite-instruction framing of
    _profile_identity_block (which would read oddly inside a review task)."""
    if not profile:
        return ""

    description = (profile.get("description") or "").strip()
    line = f'\nTarget writing profile: "{profile["name"]}"'
    if description:
        line += f" — {description}"
    return line


def _tokenize(text: str) -> set:
    return {
        w for w in re.findall(r"[a-z']+", text.lower())
        if len(w) > 2 and w not in _STOPWORDS
    }


def _relevance_score(content_tokens: set, phrase: str) -> int:
    phrase_tokens = _tokenize(phrase)
    if not phrase_tokens:
        return 0
    return len(content_tokens & phrase_tokens)


def _select_relevant_phrases(content: str, trained: List[dict]) -> List[dict]:
    """Rank trained phrases by simple token-overlap relevance to the input
    content (no vector DB — just shared content words), so the few-shot
    examples shown to the model are the ones most likely to actually apply,
    instead of 20 recency-ordered examples that may have nothing to do with
    this particular input.

    If fewer than MIN_FEWSHOT_EXAMPLES score above zero, pad with the most
    recent phrases (trained is already recency-ordered — see
    db.list_trained_phrases()) so a sparsely-trained profile still gets some
    style signal rather than none.
    """
    if not trained:
        return []

    content_tokens = _tokenize(content)
    scored = [(_relevance_score(content_tokens, t["ai_phrase"]), t) for t in trained]
    scored.sort(key=lambda pair: pair[0], reverse=True)

    relevant = [t for score, t in scored if score > 0][:MAX_FEWSHOT_EXAMPLES]

    target_floor = min(MIN_FEWSHOT_EXAMPLES, len(trained))
    if len(relevant) < target_floor:
        seen_ids = {t["id"] for t in relevant}
        for t in trained:
            if len(relevant) >= target_floor:
                break
            if t["id"] not in seen_ids:
                relevant.append(t)
                seen_ids.add(t["id"])

    return relevant


def _style_guidelines_block(examples: List[dict]) -> str:
    if not examples:
        return ""
    lines = [
        "\n--- USER WRITING STYLE & TRAINED PHRASE GUIDELINES ---",
        "The user has provided the following examples of how they prefer AI-sounding phrases to be rewritten into their personal voice.",
        "These are the examples most relevant to the text you're rewriting. Use them to infer the user's preferred vocabulary, tone, style, and sentence structure. Adapt the input text dynamically using these style cues:\n",
    ]
    for idx, t in enumerate(examples, start=1):
        lines.append(
            f'Example {idx}:\n'
            f'  AI-sounding phrase: "{t["ai_phrase"].strip()}"\n'
            f'  User\'s preferred style: "{t["humanized_phrase"].strip()}"'
        )
    lines.append("--- END OF STYLE GUIDELINES ---")
    return "\n" + "\n".join(lines)


def _call_groq(system_prompt: str, user_content: str, temperature: float = 0.7) -> str:
    client = get_client()
    model = os.environ.get("GROQ_MODEL", DEFAULT_MODEL)
    max_tokens = min(4096, max(512, int(len(user_content.split()) * 2.5)))

    completion = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        # gpt-oss models spend an unpredictable chunk of the token budget on
        # hidden reasoning before emitting visible content; without capping
        # that effort, a short max_tokens budget can get eaten entirely by
        # reasoning and truncate the actual output mid-sentence.
        reasoning_effort="low",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    )
    return completion.choices[0].message.content.strip()


def _quality_check_pass(original_content: str, rewrite: str, profile: Optional[dict]) -> str:
    """Second, lightweight LLM pass over the rewrite: catches and minimally
    repairs genuine defects (same-root repetition, grammar errors, meaning
    drift, tone that contradicts the profile) without rewriting wholesale.

    Fails safe: any error here — API failure, empty response, anything —
    just returns the original `rewrite` untouched rather than losing it.
    """
    try:
        system_prompt = QUALITY_CHECK_SYSTEM_PROMPT + _profile_context_line(profile)
        user_content = f"ORIGINAL:\n{original_content}\n\nREWRITE TO CHECK:\n{rewrite}"
        checked = _call_groq(system_prompt, user_content, temperature=0.3)
        return checked if checked.strip() else rewrite
    except Exception:
        return rewrite


def humanize_content(
    content: str,
    profile_id: Optional[int] = None,
    phrase_ids: Optional[List[int]] = None,
) -> str:
    profile = get_profile(profile_id) if profile_id is not None else None

    trained = list_trained_phrases(profile_id=profile_id)
    if not trained and profile_id is not None:
        # Fallback to all trained phrases if profile has no phrases defined
        trained = list_trained_phrases()

    if phrase_ids:
        # User hand-picked specific phrases — use exactly those, in the
        # order they were picked, instead of the relevance ranking below.
        by_id = {t["id"]: t for t in trained}
        relevant_examples = [by_id[pid] for pid in phrase_ids if pid in by_id]
    else:
        relevant_examples = _select_relevant_phrases(content, trained)

    system_prompt = (
        BASE_GUIDELINES
        + _profile_identity_block(profile)
        + _style_guidelines_block(relevant_examples)
        + '\n\nReturn ONLY the rewritten text, nothing else. No preamble, no markdown headers, no quotation marks wrapping the whole output.'
    )
    rewrite = _call_groq(system_prompt, content)
    checked = _quality_check_pass(content, rewrite, profile)
    return scrub_ai_signals(checked)
