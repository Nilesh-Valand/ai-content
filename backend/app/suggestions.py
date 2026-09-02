"""
LLM-powered suggestion generation via Groq.

IMPORTANT: This module never computes or influences the AI-likelihood
score. It only takes the already-computed rule-based signals and asks
the LLM to phrase natural-language improvement suggestions, in the same
style as the product spec's example:

    Detected sentence: "AI is revolutionizing the modern business landscape."
    Suggestion: This statement is broad and generic. Explain exactly how...
    Improved direction: Companies are using AI assistants to handle...
"""
import json
import os
from typing import List

from groq import Groq

from .models import DetectedPattern, Suggestion, SentenceScore

_client = None

# Must be a model currently served by the account's Groq API key — models get
# deprecated/removed over time and a stale name here fails every request with
# a 404 NotFoundError. Check what's actually available with client.models.list().
DEFAULT_MODEL = "openai/gpt-oss-120b"


def get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set")
        _client = Groq(api_key=api_key)
    return _client


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
    model = os.environ.get("GROQ_MODEL", DEFAULT_MODEL)

    try:
        completion = client.chat.completions.create(
            model=model,
            temperature=0.4,
            max_tokens=1500,
            reasoning_effort="low",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(payload)},
            ],
        )
        raw = completion.choices[0].message.content.strip()
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
                           f"Try again or check GROQ_API_KEY / model name.",
                improved_direction=None,
            )
        ]
