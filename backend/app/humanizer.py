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
  4. A deterministic rule-based scrub (rule_scrubber.py) that removes
     whichever of the detector's own flaggable surface patterns survived —
     AI vocabulary, sentence-opener crutches, em dash overuse, stray
     emoji — since this app owns the detector too and can target its exact
     signals directly instead of hoping the LLM avoided them all.
  5. A closed-loop detector-feedback pass: the same rule-based analyzer
     this app uses to score AI-writing likelihood (analyzer.py) is run
     against the humanized output. If it still scores above the target
     threshold, the specific signals still firing (low sentence-length
     variation, leftover AI vocabulary, low specificity, negative
     parallelism, etc. — the same surface families independently
     documented as what third-party detectors such as GPTZero, ZeroGPT,
     Originality.ai and Copyleaks key on) are turned into targeted revision
     instructions and fed back to the LLM for a minimal, meaning-preserving
     revision. This repeats for a bounded number of rounds, keeping
     whichever candidate actually scored lowest.
"""
import os
import re
import time
from typing import List, Optional, Tuple

import groq

from .analyzer import analyze_text, get_nlp
from .db import get_profile, list_trained_phrases
from .perplexity import score_document as score_perplexity
from .rule_scrubber import scrub_ai_signals
from .suggestions import DEFAULT_MODEL, get_client

BASE_GUIDELINES = """You are an expert human editor and personalized writing assistant. Rewrite the given text so it reads as naturally human-written while matching the user's personal writing voice and preserving the original meaning, facts, tone, and approximate length.

Guidelines:
- Preserve the input's paragraph structure exactly: the same number of paragraphs, in the same order, separated by the same blank lines. Never merge multiple input paragraphs into one, and never split the output into a single dense block — a wall of text with no paragraph breaks is itself a strong AI tell.
- Vary paragraph length the way real writing does — some paragraphs should be one or two sentences, others four or five. Uniform, similarly-sized paragraphs back to back are as much of an AI tell as uniform sentence length.
- Vary sentence length dramatically within the same paragraph — mix several short, punchy sentences (5-10 words) with longer, more complex ones. Uniform sentence length is one of the strongest tells of AI-generated text ("low burstiness"); avoid it.
- Vary how sentences open. Never start more than one sentence in the same paragraph with the same word, and avoid leading with stock transition words (Furthermore, Moreover, Additionally, However, Therefore, In conclusion, Overall, etc.) altogether.
- Never construct a three-item parallel list, whether it's formatted as a bulleted list, written inline with "and"/"or" ("enhancing X, fostering Y, and driving Z"), or strung together with no conjunction at all ("apps can be lifesavers, calendars, task trackers, reminders"). All three shapes are the same "rule of three" listicle cadence, and it's one of the most well-documented AI tells there is. Use two items, four items, or a single plainly stated point instead of exactly three in a row.
- Avoid turning individual points into short, tidy, quotable "life advice" one-liners ("We all have the same 24 hours," "Skip the to-do list that never gets done") — that clean, self-contained maxim cadence is itself a hallmark of AI-generated self-help and listicle content, independent of the specific words used, and it's exactly the shape real detectors flag most reliably. Instead: tie the point to something more specific or situational, let it run on into a slightly longer and looser sentence with concrete detail rather than a tidy compressed one, or fold it into the sentence before or after it instead of giving it its own standalone beat.
- Never use "not just X, but Y" or "it's not X, it's Y" constructions (negative parallelism) — state the point directly.
- Do not use em dashes (—). Use a period, comma, or parentheses instead.
- Do not use emoji, hashtags, or markdown formatting (no asterisks, headers, or bullet lists) anywhere in the output.
- Prefer concrete, natural, and expressive phrasing over generic AI filler words and hedge phrases (e.g. "it's worth noting," "in today's fast-paced world," "at the end of the day," "when it comes to").
- Write with a genuine human voice, not a smoothed-out summary of one: it's fine for a sentence to trail into a related thought with a comma, for a paragraph to open on a small aside, or for phrasing to be slightly informal where the register allows it — real writing isn't uniformly polished the way AI output tends to be.
- Avoid the reflex of picking the single most predictable next word. AI-generated prose tends toward the statistically "safest," most expected phrasing at every turn; real human writing takes small, natural detours — an unexpected but apt word choice, a mildly colloquial turn of phrase, a bit of a specific idiom — instead of the smoothest possible path through the sentence. Lean into that instead of optimizing every clause for maximum polish.
- Don't make every sentence maximally information-dense. Real writers let a thought breathe sometimes, restate something slightly differently for emphasis, or add a small aside that isn't strictly necessary — a piece where every single sentence is packed with new information and zero redundancy reads as machine-written, not human-written.
- It's fine, and often more natural, for a sentence to be structurally imperfect in a human way — a fragment used for emphasis, a clause tacked on with "and" or "but" that a strict editor might trim, a slightly looser structure than the tightest possible phrasing — as long as it stays grammatically sound and clear.
- Genuinely restructure at the sentence level, not just at the word level — swapping synonyms into the same sentence shape isn't enough. Within a paragraph: combine two of the original's shorter sentences into one where it reads naturally, split one of its longer sentences into two, reorder a clause (put the condition or the consequence first instead of always following the original's order), or turn a plain statement into a brief rhetorical question the next sentence answers. The goal is prose whose sentence-by-sentence shape doesn't mirror the input's, even though every point in it still does.
- Vary register slightly across the piece the way a real person's writing does — not everything needs the same level of formality throughout; a stray moment of directness, a short aside to the reader, or a comparison/analogy in one place and not another is normal and human, even in a fairly neutral piece.
- Where the original already contains a number, name, date, or specific example, keep it intact and prominent rather than smoothing it into a vague generality — that concreteness is what makes prose read as human-written.
- Never invent a new number, statistic, percentage, date, or named person/company/study that is not in the original text, even if the prose would otherwise read as generic — specificity is only ever surfaced from the source, never fabricated.
- Use natural contractions (it's, don't, you're, that's) where the register of the piece allows it, the way a real person actually writes.
- Keep the same point of view, register, and intent as the original.
- Do not add new facts, claims, or details that were not implied by the original.
- Do not add commentary, notes, or explanations about the rewrite.
- Do NOT perform rigid, literal string substitution. Apply style preferences flexibly and intelligently according to surrounding context.
- Do NOT summarize, condense, or shorten the text. Rewrite it sentence by sentence and paragraph by paragraph, covering every point in the original — the rewrite must contain roughly the same number of sentences and paragraphs as the input, not a shorter recap of it.
- A "style" or "platform" instruction (e.g. Instagram, LinkedIn) changes tone, vocabulary, and voice only — it is never a license to cut content or turn a full passage into a single short line or caption."""

# --- Closed-loop detector feedback -----------------------------------------
#
# Human-readable, actionable guidance for each analyzer signal (see
# analyzer.py WEIGHTS) — used to turn "this signal is still elevated" into a
# concrete instruction the LLM can act on for a targeted revision, rather
# than a vague "try harder" re-prompt.
_SIGNAL_GUIDANCE = {
    "ai_vocab": "Replace any remaining generic AI-marketing vocabulary or buzzwords (e.g. \"robust\", \"seamless\", \"leverage\", \"delve\", \"underscores\") with plain, specific wording.",
    "generic_promo": "Replace any remaining generic promotional filler (\"best-in-class\", \"industry-leading\", \"actionable insights\", etc.) with plain, concrete wording.",
    "predictability": "Vary sentence openers — no two sentences in the same paragraph should start with the same word, and drop stock transition words (however, therefore, furthermore, additionally, notably, overall, in conclusion) as sentence openers entirely.",
    "repetition": "Remove repeated word pairs or three-word phrases — the same short sequence of words appears more than once; reword one of the occurrences.",
    "diversity": "Increase word variety — several words or word-roots are being reused too often; use different phrasing where the same idea recurs.",
    "rule_of_three": "Break up any \"X, Y, and Z\" three-item parallel list into flowing prose, or restructure it to two or four items instead of exactly three.",
    "sentence_variation": "Vary sentence length much more — the current sentences are too uniform in length. Mix short (5-10 word) sentences with longer, more complex ones in the same paragraph.",
    "specificity": "If the ORIGINAL SOURCE (given below) contains a number, name, date, or specific example that got smoothed away in the current text, restore it. Do NOT invent a new number, statistic, percentage, date, or named person/company/study that isn't in the original source — if the original has no such detail to draw on, leave this issue alone rather than fabricating one.",
    "negative_parallelism": "Remove any \"not just X, but Y\" or \"it's not X, it's Y\" construction — state the point directly instead.",
    "em_dash_overuse": "Remove em dashes (—) entirely — replace each with a period, comma, or parentheses.",
    "perplexity": "The wording and sentence rhythm still read as too statistically predictable to a language-model-based check, even though no specific banned phrase is present — this is the actual perplexity/burstiness signal real AI detectors use, distinct from any surface wordlist. Make bolder, less obvious word choices instead of the safest synonym in several places, and push sentence length and rhythm to vary even more sharply from one sentence to the next within each paragraph.",
}

# Based on real per-sentence flagging from a third-party detector on one of
# this app's own outputs: every flagged sentence was a short, tidy,
# self-contained "life advice" maxim ("We all have the same 24 hours"),
# while short reactive fragments and long messy concrete enumerations both
# went unflagged. No rule-based signal can detect this pattern (it's a
# genre/tone cadence, not a specific word or structure), so it can't gate a
# revision round on its own the way the signals above do — instead it
# rides along as a standing reminder whenever a round is already happening
# for another reason (see the loop in _refine_against_detector).
_APHORISM_REMINDER = (
    "Check for short, tidy, standalone \"life advice\" one-liners (a complete, quotable "
    "generalization like \"We all have the same 24 hours\") and rewrite any you find so the "
    "point is tied to something more specific or situational, or folded into a longer, "
    "looser sentence — that clean maxim cadence is a strong AI tell independent of wording."
)

# A signal below this is not worth spending a revision instruction on.
_SIGNAL_FEEDBACK_THRESHOLD = 0.25

# Stop refining once the analyzer's own overall score is at/below this. Set
# well under the analyzer's own "Low" cutoff (30) for headroom against
# third-party detectors, which aren't guaranteed to weight signals the same
# way this app's analyzer does.
_TARGET_SCORE_PCT = 15.0

# Even once the weighted overall score clears the target above, a single
# glaring signal (e.g. a leftover rule-of-three list) can still be exactly
# the kind of thing a human reader — or a different detector weighting
# signals differently — would flag on its own. Keep refining until every
# individual signal is under this stricter per-signal ceiling too.
_SIGNAL_HARD_CAP = 0.45

# Bounded number of extra LLM revision rounds after the first draft — keeps
# latency/cost predictable while still giving the closed loop real room to
# converge on a low score. Each round is a full Groq call carrying the
# entire ORIGINAL SOURCE text, so this is also the main lever on how much a
# single /humanize request draws down the account's token budget (a real,
# hard constraint — see the token-per-day quota note on _quality_check_pass
# being deferred to one pass total rather than one per round). Kept
# minimal since most inputs converge within a single round anyway (see
# _refine_against_detector's early-exit check), and a request that still
# isn't converging after one targeted revision is unlikely to be fixed by
# brute-forcing further rounds regardless.
_MAX_REFINEMENT_ROUNDS = 1

REVISION_SYSTEM_PROMPT = """You are revising a piece of text that a rule-based AI-writing detector still flags as likely AI-generated. You will be given the ORIGINAL SOURCE text (for reference only — do not reintroduce its exact generic phrasing), the CURRENT text to revise, and a specific list of issues the detector found in the CURRENT text.

Make the smallest edits needed to fix exactly those issues. Preserve the original meaning, facts, tone, point of view, approximate length, and paragraph structure (same number of paragraphs, same blank-line breaks, in the same order) — this is a targeted revision, not a rewrite from scratch. Do not introduce new AI-sounding phrasing while fixing one issue. Do not add commentary, notes, or explanations. Do not use emoji, hashtags, or markdown formatting.

Never invent a new number, statistic, percentage, date, or named person/company/study that is not present in the ORIGINAL SOURCE, even when an issue asks you to add specificity — only surface concrete details that are genuinely already in the source material.

Return ONLY the revised text, nothing else."""


def _build_signal_feedback(signals: dict) -> List[str]:
    """Turns the analyzer's per-signal scores into a short list of concrete,
    actionable revision instructions — only for signals still elevated
    enough to be worth a targeted fix."""
    return [
        _SIGNAL_GUIDANCE[key]
        for key, score in sorted(signals.items(), key=lambda kv: kv[1], reverse=True)
        if score > _SIGNAL_FEEDBACK_THRESHOLD and key in _SIGNAL_GUIDANCE
    ]

MIN_FEWSHOT_EXAMPLES = 3
MAX_FEWSHOT_EXAMPLES = 8

QUALITY_CHECK_SYSTEM_PROMPT = """You are a meticulous copy editor doing a final quality pass on an already-rewritten piece of text. You will be shown the ORIGINAL text and a REWRITE of it. Your only job is to catch and fix genuine defects in the REWRITE:
- Same-root word repetition or awkward wordplay (e.g. "spotting the spots", "using the use of")
- Grammar or punctuation errors
- Meaning drift — the rewrite says something different from the original
- Fabricated specifics — a number, statistic, percentage, date, or named person/company/study that appears in the REWRITE but is NOT present anywhere in the ORIGINAL. This is a serious defect: remove or generalize the fabricated detail back to what the ORIGINAL actually supports.
- Paragraph breaks collapsed or merged compared to the ORIGINAL — if the ORIGINAL had multiple paragraphs and the REWRITE flattened them into fewer (or one dense block), restore the original paragraph breaks at the equivalent points in the REWRITE.
- Tone that contradicts the target writing profile, if one is given below

Rules:
- Make the SMALLEST possible edit to fix a real problem. Do not rewrite wholesale.
- If the REWRITE has no genuine defects, return it completely unchanged, word for word.
- Never change the meaning, add new information, or significantly change the length.
- A deliberate stylistic choice is not a defect: a short sentence fragment used for emphasis, a clause tacked on with "and" or "but," a slightly loose or informal structure, or mild restatement for emphasis are normal features of human writing, not errors to tidy up. Only fix what's actually wrong — don't smooth the text into more uniform, "textbook-correct" prose than it needs to be.
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


# The closed-loop refinement below can make anywhere from ~3 to ~10+ Groq
# calls for a single /humanize request (initial rewrite, guards, quality
# checks, up to _MAX_REFINEMENT_ROUNDS revision rounds each with their own
# quality check). That volume makes hitting Groq's own rate limit far more
# likely than a simple one-shot call would, so every call through here
# retries on a 429 instead of letting it bubble straight up into a failed
# request — a transient rate limit should cost the user a few seconds of
# extra wait, not the whole humanize call.
_MAX_RATE_LIMIT_RETRIES = 3
_RATE_LIMIT_MAX_WAIT_SECONDS = 15.0

# Conservative estimate of what one more Groq call is likely to cost
# (prompt + response tokens). Optional calls — guard retries, extra
# refinement rounds, the final quality check — check _has_token_headroom
# against this before firing, so a tight per-minute budget (seen as low as
# 8000 tokens/min on some keys) gets spent on the calls that matter most
# (the initial rewrite always happens) rather than risking a failed call
# partway through optional polishing.
_MIN_TOKEN_HEADROOM = 2500


def _rate_limit_wait_seconds(error: "groq.RateLimitError", attempt: int) -> float:
    """Honors the API's own Retry-After header when present (it knows
    exactly when the limit resets); falls back to exponential backoff
    (2s, 4s, 8s, ...) otherwise. Either way, capped so one retry can't
    stall a request for an unreasonable amount of time."""
    try:
        header = error.response.headers.get("retry-after")
        if header is not None:
            return min(float(header), _RATE_LIMIT_MAX_WAIT_SECONDS)
    except Exception:
        pass
    return min(2.0 * (2 ** attempt), _RATE_LIMIT_MAX_WAIT_SECONDS)


# Live rate-limit headroom, updated from Groq's own response headers after
# every successful call (x-ratelimit-remaining-tokens — see _call_groq).
# None until the first call completes, meaning "unknown" rather than "OK":
# callers that check this fail open on None, since a missing header
# shouldn't itself block anything. This is deliberately process-global,
# single-value state rather than anything more elaborate — good enough for
# a same-process, mostly-sequential pipeline to avoid attempting optional
# calls it can already see will fail, not a precise multi-worker tracker.
_last_remaining_tokens: Optional[int] = None


def _update_rate_limit_state(headers) -> None:
    global _last_remaining_tokens
    try:
        remaining = headers.get("x-ratelimit-remaining-tokens")
        if remaining is not None:
            _last_remaining_tokens = int(remaining)
    except Exception:
        pass


def _has_token_headroom(min_tokens: int) -> bool:
    """True if the last known remaining-token count leaves room for
    another call of roughly this size, or if it's unknown (fail open —
    a missing/unparseable header shouldn't itself block a call)."""
    return _last_remaining_tokens is None or _last_remaining_tokens >= min_tokens


# Groq scopes its rate limits (including the daily token quota) per model,
# not per account — a model that's hit its own daily cap can sit right
# next to other models on the exact same key/account that still have full
# headroom, because none of today's usage has touched them. When the
# primary model's own retries are exhausted on a rate limit, _call_groq
# falls back to this one rather than failing the request outright. Kept in
# the same "openai/gpt-oss" family as the default primary model (see
# suggestions.DEFAULT_MODEL) as the closest match in how it's likely to
# follow this app's prompts, just a smaller variant.
_FALLBACK_MODEL = os.environ.get("GROQ_FALLBACK_MODEL", "openai/gpt-oss-20b")


def _call_groq(system_prompt: str, user_content: str, temperature: float = 0.7) -> str:
    client = get_client()
    primary_model = os.environ.get("GROQ_MODEL", DEFAULT_MODEL)
    max_tokens = min(4096, max(512, int(len(user_content.split()) * 2.5)))

    models_to_try = [primary_model]
    if _FALLBACK_MODEL and _FALLBACK_MODEL != primary_model:
        models_to_try.append(_FALLBACK_MODEL)

    last_error: Optional[Exception] = None
    for model_idx, model in enumerate(models_to_try):
        is_last_model = model_idx == len(models_to_try) - 1
        # Retrying the SAME model on a rate limit only makes sense when
        # there's no fallback to move to instead — a model that's hit its
        # own daily quota won't recover within a short backoff window
        # regardless of how many times it's retried, so when another
        # model is available it's tried immediately rather than burning
        # up to _MAX_RATE_LIMIT_RETRIES backoff waits first. Retries are
        # reserved for transient (e.g. per-minute burst) limits on
        # whichever model is the last one left to try.
        retries_for_this_model = _MAX_RATE_LIMIT_RETRIES if is_last_model else 0
        for attempt in range(retries_for_this_model + 1):
            try:
                # with_raw_response gives access to Groq's rate-limit
                # headers (remaining tokens/requests, reset time) alongside
                # the normal parsed completion — used to let the closed
                # loop see its own remaining budget and back off before
                # hitting the limit, rather than only ever finding out by
                # way of a failed call.
                response = client.chat.completions.with_raw_response.create(
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    # gpt-oss models spend an unpredictable chunk of the
                    # token budget on hidden reasoning before emitting
                    # visible content; without capping that effort, a
                    # short max_tokens budget can get eaten entirely by
                    # reasoning and truncate the actual output mid-sentence.
                    reasoning_effort="low",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                )
                _update_rate_limit_state(response.headers)
                completion = response.parse()
                return completion.choices[0].message.content.strip()
            except groq.RateLimitError as e:
                last_error = e
                if attempt >= retries_for_this_model:
                    break  # give up on this model; try the next one, if any
                time.sleep(_rate_limit_wait_seconds(e, attempt))

    raise last_error


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


# Below this fraction of the original word count, a rewrite is treated as an
# accidental summary (the model condensed instead of rewrote) rather than a
# legitimately tighter phrasing, and gets one retry with a blunter prompt.
# Also used as a hard floor throughout the closed-loop refinement rounds
# below (see _length_issue) — not just the first draft — since the
# analyzer's own signals don't penalize brevity: a heavily truncated,
# choppy result can look "clean" to a surface-pattern scorer precisely
# because there's less text left for its rules to find anything wrong
# with, even though it dropped most of the source's actual content.
_MIN_LENGTH_RATIO = 0.6


def _length_ratio(original_content: str, text: str) -> float:
    original_words = len(original_content.split())
    if original_words == 0:
        return 1.0
    return len(text.split()) / original_words


def _length_issue(original_content: str, text: str) -> Optional[str]:
    """Returns a revision instruction if `text` has been cut down too far
    relative to `original_content`, else None. Only applies once the source
    is long enough that "roughly the same length" is a meaningful ask."""
    if len(original_content.split()) < 20:
        return None
    ratio = _length_ratio(original_content, text)
    if ratio >= _MIN_LENGTH_RATIO:
        return None
    return (
        f"The current text has been cut down to only about {round(ratio * 100)}% of the "
        f"ORIGINAL SOURCE's length — this is a serious problem, not a stylistic choice. "
        f"Restore the missing content: every point made in the ORIGINAL SOURCE should "
        f"still be present here, reworded but not dropped. Do not summarize."
    )


def _candidate_rank(original_content: str, text: str, score: float) -> tuple:
    """Sort key for picking the best candidate across refinement rounds.
    A candidate that dropped too much content always loses to one that
    didn't, regardless of how low its detector score is — length is a hard
    constraint here, not one more signal to trade off against the rest.
    Among length-compliant candidates, lower detector score wins; among
    non-compliant ones (worst case: every round still came back short),
    the longest — i.e. least-truncated — one wins instead."""
    ratio = _length_ratio(original_content, text)
    if ratio >= _MIN_LENGTH_RATIO:
        return (0, score)
    return (1, -ratio)


def humanize_content(
    content: str,
    profile_id: Optional[int] = None,
    phrase_ids: Optional[List[int]] = None,
) -> str:
    """Backward-compatible string-only entry point — see humanize_with_score
    for the (text, ai_score_after) pair the API surfaces to the client."""
    text, _score = humanize_with_score(content, profile_id=profile_id, phrase_ids=phrase_ids)
    return text


def humanize_with_score(
    content: str,
    profile_id: Optional[int] = None,
    phrase_ids: Optional[List[int]] = None,
) -> Tuple[str, Optional[float]]:
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
    # Higher than the _call_groq default (0.7): a close paraphrase at low
    # temperature stays anchored near the original's own token-probability
    # distribution — which is exactly what a statistical/perplexity-based
    # detector keys on, independent of any surface wording — so the first
    # draft deliberately asks for more lexical/structural risk-taking.
    rewrite = _call_groq(system_prompt, content, temperature=0.95)
    rewrite = _guard_against_summarizing(system_prompt, content, rewrite)
    rewrite = _guard_against_paragraph_collapse(system_prompt, content, rewrite)
    # No quality-check pass here — it runs once, at the very end of
    # _refine_against_detector, on whichever candidate across all rounds
    # actually wins, rather than once per round. Every Groq call in this
    # pipeline draws down the same account-level token budget, and a
    # per-round quality check was the single largest avoidable share of
    # that (as many calls as refinement rounds, on top of everything
    # else) for a check that only matters on the text actually returned.
    candidate = scrub_ai_signals(rewrite, original=content)

    return _refine_against_detector(candidate, content, profile)


# Weight given to the local perplexity/burstiness signal (perplexity.py)
# versus the rule-based analyzer's own weighted score, when blending them
# into the single overall score the closed loop optimizes against. Kept a
# minority share deliberately: the rule-based signals come with concrete,
# targeted fixes the revision prompt can act on (a specific overused
# phrase, a specific negative-parallelism construction), while perplexity
# is a diffuse "read as more/less predictable overall" signal with no
# single sentence to point at — useful as a check on the rest, less useful
# as the dominant driver of what to edit.
_PERPLEXITY_WEIGHT = 0.35


def _score_candidate(candidate: str) -> Tuple[float, dict]:
    """Runs this app's own rule-based analyzer over a candidate rewrite,
    plus (when available) the local GPT-2 perplexity/burstiness signal —
    the same statistical property ZeroGPT's own published methodology
    says it measures, which the rule-based analyzer structurally can't see
    at all (see perplexity.py). Blends the two into one overall score the
    closed loop optimizes against, and folds the perplexity signal into the
    per-key `signals` dict too, so a low-perplexity ("too predictable")
    result can generate its own targeted revision instruction just like any
    other signal.

    Fails safe: if the analyzer errors, treated as already good enough
    rather than blocking the call. If the local perplexity model isn't
    available, silently falls back to the rule-based score alone — this is
    an enhancement layer, not a hard dependency.
    """
    try:
        result = analyze_text(candidate)
        overall_pct, signals = result["overall_pct"], dict(result["signals"])
    except Exception:
        overall_pct, signals = 0.0, {}

    try:
        sentences = [s.text.strip() for s in get_nlp()(candidate).sents if s.text.strip()]
        perplexity_result = score_perplexity(sentences)
    except Exception:
        perplexity_result = None

    if perplexity_result is not None:
        _mean_ppl, _burstiness, ai_likelihood = perplexity_result
        signals["perplexity"] = ai_likelihood
        overall_pct = round(
            (1 - _PERPLEXITY_WEIGHT) * overall_pct + _PERPLEXITY_WEIGHT * (ai_likelihood * 100),
            1,
        )

    return overall_pct, signals


def _refine_against_detector(
    candidate: str, original_content: str, profile: Optional[dict]
) -> Tuple[str, Optional[float]]:
    """Closed-loop pass: scores `candidate` with this app's own analyzer —
    the same signal families (sentence-length burstiness, AI vocabulary,
    specificity, repetition, negative parallelism, em dash overuse, rule of
    three) that third-party detectors independently key on — and, while it's
    still above the target threshold (or has dropped too much content — see
    _length_issue), asks the LLM for a small targeted revision addressing
    exactly the issues still present. Keeps whichever candidate across all
    rounds ranks best by _candidate_rank, which treats content length as a
    hard floor rather than one more signal to trade off against the rest —
    so a round that "improves" the detector score by cutting more content
    can never be selected as the final output.

    Fails safe: any scoring/revision error along the way just returns the
    best candidate found so far, never raises.
    """
    best_text, best_score = candidate, None
    try:
        score, signals = _score_candidate(candidate)
        best_text, best_score = candidate, score
        best_rank = _candidate_rank(original_content, candidate, score)
        current = candidate

        for _ in range(_MAX_REFINEMENT_ROUNDS):
            length_issue = _length_issue(original_content, current)
            worst_signal = max(signals.values()) if signals else 0.0
            signals_ok = score <= _TARGET_SCORE_PCT and worst_signal <= _SIGNAL_HARD_CAP
            if signals_ok and not length_issue:
                break
            if not _has_token_headroom(_MIN_TOKEN_HEADROOM):
                # Preserve whatever budget is left rather than spend it on
                # a revision round that's likely to hit the rate limit
                # anyway — the best candidate found so far is still a
                # complete, valid result, just not as refined as it could
                # be with more headroom.
                break

            issues = _build_signal_feedback(signals)
            if length_issue:
                # Restoring dropped content takes priority over chasing
                # remaining surface signals — a shorter "cleaner" result is
                # not an improvement if it got that way by cutting content.
                issues = [length_issue] + issues
            if not issues:
                break
            # No rule-based signal can detect this one (it's a genre/tone
            # pattern, not a vocabulary or structural one), so it can't
            # gate whether a revision round happens the way the signals
            # above do — but it's cheap to remind the model of it whenever
            # a round is already happening for other reasons.
            issues.append(_APHORISM_REMINDER)

            issues_block = "\n".join(f"- {line}" for line in issues)
            revision_prompt = (
                REVISION_SYSTEM_PROMPT
                + _profile_context_line(profile)
                + f"\n\nIssues the detector found in the current text:\n{issues_block}"
            )
            revision_input = f"ORIGINAL SOURCE:\n{original_content}\n\nCURRENT TEXT:\n{current}"
            revised = _call_groq(revision_prompt, revision_input, temperature=0.6)
            # No quality-check pass here — see the note in humanize_with_score.
            # Deferred to a single pass below, on whichever candidate wins.
            revised = scrub_ai_signals(revised, original=original_content)

            score, signals = _score_candidate(revised)
            current = revised
            rank = _candidate_rank(original_content, revised, score)
            if rank < best_rank:
                best_text, best_score, best_rank = revised, score, rank

        # One quality-check pass total, on whichever candidate actually won
        # (initial draft or a later revision) — catches same-root
        # repetition, grammar errors, fabricated specifics, and collapsed
        # paragraph structure in the text that's actually being returned,
        # at a fraction of the token cost of checking after every round.
        # _quality_check_pass already fails safe on its own (falls back to
        # its input unchanged on any error), but skipping it outright when
        # headroom is already known to be too tight avoids spending the
        # retry cycle (up to _MAX_RATE_LIMIT_RETRIES attempts) on a call
        # that's very unlikely to succeed anyway.
        if _has_token_headroom(_MIN_TOKEN_HEADROOM):
            best_text = _quality_check_pass(original_content, best_text, profile)
            best_text = scrub_ai_signals(best_text, original=original_content)
        return best_text, best_score
    except Exception:
        return best_text, best_score


def _guard_against_summarizing(system_prompt: str, content: str, rewrite: str) -> str:
    """One retry, with a blunter instruction, if the model collapsed the
    input into a much shorter summary instead of rewriting it in full —
    this is what happens when a platform-style profile (e.g. Instagram)
    nudges the model toward caption-length output regardless of input size.
    Fails safe: on any error, or if the retry is no better, keeps the
    original rewrite rather than losing it."""
    original_words = len(content.split())
    rewrite_words = len(rewrite.split())
    if original_words < 20 or rewrite_words >= original_words * _MIN_LENGTH_RATIO:
        return rewrite
    if not _has_token_headroom(_MIN_TOKEN_HEADROOM):
        # A retry here would likely just fail on the rate limit anyway —
        # better to keep the imperfect rewrite than spend remaining budget
        # (and a retry cycle) on a call that probably won't succeed.
        return rewrite

    try:
        retry_prompt = (
            system_prompt
            + f"\n\nYour previous attempt dropped most of the content and returned only "
              f"{rewrite_words} words for a {original_words}-word input. That is wrong — "
              f"do not summarize. Rewrite the ENTIRE input in full, sentence by sentence, "
              f"matching its length and covering every point."
        )
        retried = _call_groq(retry_prompt, content, temperature=0.7)
        if len(retried.split()) > rewrite_words:
            return retried
        return rewrite
    except Exception:
        return rewrite


def _count_paragraphs(text: str) -> int:
    return len([p for p in re.split(r"\n\s*\n", text) if p.strip()])


# Below this fraction of the original paragraph count, a rewrite is treated
# as having flattened the input's structure (merged several paragraphs into
# one dense block) rather than legitimately combining a couple of short
# ones — a uniform wall of text is itself a recognizable AI tell, on top of
# just reading as noticeably shorter/denser than the original.
_MIN_PARAGRAPH_RATIO = 0.7


def _guard_against_paragraph_collapse(system_prompt: str, content: str, rewrite: str) -> str:
    """One retry, with an explicit paragraph-count instruction, if the model
    merged the input's paragraphs into a single (or far fewer) block(s) —
    this happens even when the word count guard above doesn't trigger,
    since collapsing structure doesn't necessarily drop words, just the
    breaks between them. Fails safe: on any error, or if the retry doesn't
    actually restore more paragraphs, keeps the original rewrite."""
    original_paras = _count_paragraphs(content)
    rewrite_paras = _count_paragraphs(rewrite)
    if original_paras < 3 or rewrite_paras >= original_paras * _MIN_PARAGRAPH_RATIO:
        return rewrite
    if not _has_token_headroom(_MIN_TOKEN_HEADROOM):
        return rewrite

    try:
        retry_prompt = (
            system_prompt
            + f"\n\nYour previous attempt merged the input's {original_paras} paragraphs down "
              f"to {rewrite_paras}. That is wrong — preserve the paragraph structure. Return "
              f"exactly {original_paras} paragraphs, in the same order, each separated by a "
              f"blank line, matching the input's structure one-for-one."
        )
        retried = _call_groq(retry_prompt, content, temperature=0.7)
        if _count_paragraphs(retried) > rewrite_paras:
            return retried
        return rewrite
    except Exception:
        return rewrite
