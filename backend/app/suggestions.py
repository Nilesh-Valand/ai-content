"""
LLM-powered suggestion generation, via either Groq or any other
OpenAI-compatible endpoint (self-hosted or third-party).

IMPORTANT: This module never computes or influences the AI-likelihood
score. It only takes the already-computed rule-based signals and asks
the LLM to phrase natural-language improvement suggestions, in the same
style as the product spec's example:

    Detected sentence: "AI is revolutionizing the modern business landscape."
    Suggestion: This statement is broad and generic. Explain exactly how...
    Improved direction: Companies are using AI assistants to handle...

--- Provider selection --------------------------------------------------

This app talks to whichever LLM provider is configured via env vars, so it
isn't locked to Groq specifically:

  - LLM_BASE_URL set  -> any OpenAI-compatible endpoint (self-hosted model,
    a third-party gateway, etc.), reached with the standard `openai`
    client. LLM_API_KEY supplies the key; LLM_MODEL supplies the model
    name (e.g. a locally-hosted "qwen3-14b").
  - LLM_BASE_URL unset -> Groq's own hosted API (the original behavior),
    reached with the `groq` client. GROQ_API_KEY / GROQ_MODEL as before.

The two client SDKs aren't interchangeable for a custom base_url: the groq
SDK hardcodes an internal "/openai/v1/..." path suffix onto whatever
base_url it's given, which only matches Groq's own API shape -- pointed at
a different OpenAI-compatible server, every request 405s. The plain
`openai` SDK doesn't add that suffix, so it works generically. Once
Groq-hosted models are also reachable as plain OpenAI-compatible
endpoints this distinction can go away, but for now the provider in use
determines which SDK backs get_client().
"""
import json
import os
import re
from typing import List, Optional

from .models import DetectedPattern, Suggestion, SentenceScore

_client = None

# Must be a model currently served by whichever provider is configured —
# models get deprecated/removed over time and a stale name here fails
# every request with a 404 NotFoundError. For Groq, check what's actually
# available with client.models.list().
DEFAULT_MODEL = "openai/gpt-oss-120b"


def using_custom_endpoint() -> bool:
    """True when a non-Groq OpenAI-compatible endpoint is configured."""
    return bool(os.environ.get("LLM_BASE_URL"))


def get_model() -> str:
    if using_custom_endpoint():
        return os.environ.get("LLM_MODEL") or DEFAULT_MODEL
    return os.environ.get("GROQ_MODEL", DEFAULT_MODEL)


def get_client():
    global _client
    if _client is None:
        base_url = os.environ.get("LLM_BASE_URL")
        if base_url:
            api_key = os.environ.get("LLM_API_KEY") or os.environ.get("GROQ_API_KEY")
            if not api_key:
                raise RuntimeError("LLM_API_KEY (or GROQ_API_KEY) is not set")
            import openai
            _client = openai.OpenAI(api_key=api_key, base_url=base_url)
        else:
            api_key = os.environ.get("GROQ_API_KEY")
            if not api_key:
                raise RuntimeError("GROQ_API_KEY is not set")
            from groq import Groq
            _client = Groq(api_key=api_key)
    return _client


def reasoning_kwargs() -> dict:
    """Extra request kwargs that keep a call's response to just the final
    answer, tuned per provider:
      - Groq's gpt-oss models: `reasoning_effort` caps how much of the
        token budget goes to hidden reasoning before visible output.
      - A "thinking" model like Qwen3 on a custom endpoint: reasoning
        isn't hidden by the API by default at all — it's emitted inline in
        the message content, wrapped in <think>...</think> tags, which
        would otherwise leak straight into whatever calls this (a
        rewrite, a suggestion, etc.). `chat_template_kwargs.enable_thinking
        = False` is the standard vLLM/Open WebUI-style switch to turn
        that off — confirmed against this app's configured endpoint:
        with it, a trivial prompt used 5 completion tokens and no <think>
        tags; without it, the same prompt used 1973 tokens almost
        entirely on an emitted reasoning trace.
    strip_thinking() below is a second, defensive layer for whenever a
    model emits <think> tags anyway despite this setting.
    """
    if using_custom_endpoint():
        return {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
    return {"reasoning_effort": "low"}


_THINK_BLOCK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_thinking(text: str) -> str:
    """Removes a leading <think>...</think> reasoning block some models
    emit inline in their response content (see reasoning_kwargs above).
    Safe no-op if there's nothing to strip."""
    if not text:
        return text
    return _THINK_BLOCK_PATTERN.sub("", text).strip()


SYSTEM_PROMPT = """You are an assistant that rewrites AI-detector findings into
short, practical improvement suggestions for a human writer. You will be given
a list of flagged sentences and the pattern(s) that triggered on each one.

For EACH flagged sentence, return an object with:
- "detected_sentence": the original sentence text
- "issue": one short sentence naming what's generic/robotic about it
- "suggestion": one short actionable instruction on how to fix it
- "improved_direction": a brief example of a more specific, natural rewrite
  (do not just restate the original sentence with synonyms — actually make
  it concrete, e.g. add a plausible specific use case, but keep it generic
  enough to not fabricate facts about a real company)

Return ONLY a JSON array of these objects, nothing else. No markdown, no
preamble, no code fences."""


def build_user_prompt(flagged_sentences: List[dict]) -> str:
    return json.dumps({"flagged_sentences": flagged_sentences}, ensure_ascii=False)


def generate_suggestions(
    sentence_scores: List[SentenceScore],
    detected_patterns: List[DetectedPattern],
    top_n: int = 5,
) -> List[Suggestion]:
    # Pick the most-flagged sentences (highest rule-based likelihood) to send to the LLM
    flagged = sorted(sentence_scores, key=lambda s: s.ai_likelihood, reverse=True)[:top_n]
    flagged = [s for s in flagged if s.ai_likelihood >= 30]

    if not flagged:
        return []

    active_patterns = [p.pattern for p in detected_patterns if p.severity in ("Medium", "High")]

    payload = [
        {
            "sentence": s.text,
            "ai_likelihood": s.ai_likelihood,
            "relevant_patterns": active_patterns,
        }
        for s in flagged
    ]

    client = get_client()
    model = get_model()

    try:
        completion = client.chat.completions.create(
            model=model,
            temperature=0.4,
            max_tokens=1500,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(payload)},
            ],
            **reasoning_kwargs(),
        )
        raw = strip_thinking(completion.choices[0].message.content.strip())
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        parsed = json.loads(raw)
        return [Suggestion(**item) for item in parsed]
    except Exception as e:
        # Fail gracefully: return a generic fallback rather than breaking the whole response
        return [
            Suggestion(
                detected_sentence=flagged[0].text if flagged else None,
                issue="Could not generate AI-powered suggestions.",
                suggestion=f"Suggestion generation failed ({type(e).__name__}). "
                           f"Try again or check the LLM provider's API key / model name.",
                improved_direction=None,
            )
        ]
