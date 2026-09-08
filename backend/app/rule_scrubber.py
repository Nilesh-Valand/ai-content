"""
Deterministic, rule-based post-processing pass over an already-LLM-humanized
text — the "hybrid" half of the humanize pipeline.

Why this exists: the LLM pass (humanizer.py) produces genuinely natural
paraphrasing, which rules alone can't do well. But because this app also
owns the detector (analyzer.py), we know exactly which surface patterns it
flags — so instead of hoping the LLM avoided all of them, this module goes
back over its output and deterministically removes the ones that are safe
to fix without any risk of changing meaning or breaking grammar:

  - AI-associated vocabulary / generic promotional phrases -> plain synonyms
  - Sentence-opener "crutch" transitions ("Furthermore, ...") -> removed
  - Em dash overuse -> replaced with commas

Signals that are NOT touched here on purpose, because a safe rule-based fix
doesn't exist for them: rule-of-three restructuring, sentence-length
variation, lexical diversity, repeated-bigram removal, negative parallelism,
and — most importantly — "lack of specificity", which would require
inventing facts that were never in the original text.

Fails safe: any unexpected error here returns the input text unchanged
rather than risking mangled output.
"""
import re
from typing import Dict

from .analyzer import get_nlp
from .wordlists import AI_ASSOCIATED_VOCAB, GENERIC_PROMOTIONAL, SENTENCE_OPENER_CRUTCHES

# Plain, natural replacements for every phrase the detector flags as
# AI-associated vocabulary or generic promotional filler. Deliberately terse
# and unremarkable — the goal is to disappear, not to sound clever.
VOCAB_REPLACEMENTS: Dict[str, str] = {
    "revolutionizing": "changing", "revolutionize": "change",
    "pivotal role": "important role", "pivotal": "key",
    "transformative": "big", "leverage": "use", "leveraging": "using",
    "streamline": "simplify", "streamlining": "simplifying",
    "seamless": "smooth", "seamlessly": "smoothly", "robust": "strong",
    "cutting-edge": "modern", "state-of-the-art": "modern",
    "in today's fast-paced world": "these days", "in today's digital age": "these days",
    "unlock the potential": "make the most", "unlocking": "opening up",
    "landscape": "world", "ever-evolving": "always changing",
    "dynamic landscape": "changing world", "paradigm shift": "big change",
    "holistic": "complete", "synergy": "teamwork", "synergies": "teamwork",
    "delve into": "look into", "delve": "dig into",
    "navigate the complexities": "deal with the challenges", "navigating": "dealing with",
    "underscore": "show", "underscores": "shows", "underscoring": "showing",
    "testament to": "proof of", "it is important to note": "keep in mind",
    "it's worth noting": "note that", "furthermore": "also", "moreover": "also",
    "in conclusion": "overall", "in summary": "overall",
    "harness the power": "make use", "harnessing": "using",
    "elevate": "improve", "elevating": "improving",
    "empower": "help", "empowering": "helping", "empowers": "helps",
    "foster": "build", "fostering": "building", "fosters": "builds",
    "bespoke": "custom", "tailored solutions": "custom options",
    "game-changer": "big deal", "game changing": "big",
    "innovative solutions": "new ideas", "meticulously": "carefully",
    "myriad": "many", "plethora": "a lot of", "top-notch": "great",
    "unparalleled": "unmatched", "unprecedented": "unusual",
    "boasts a": "has a", "boasts": "has", "bolstered": "strengthened",
    "crucial": "key", "emphasizing": "stressing", "enduring": "lasting",
    "garnered": "got", "garner": "get", "intricate": "detailed",
    "intricacies": "details", "interplay": "connection", "tapestry": "mix",
    "vibrant": "lively", "align with": "match", "aligns with": "matches",
    "resonates with": "connects with", "enhancing": "improving", "enhance": "improve",
    "highlighting": "showing", "showcasing": "showing", "showcases": "shows",
    "showcase": "show", "deep dive": "close look", "exemplifies": "shows",
    "profound": "deep", "commitment to": "focus on",
    "stands as a testament": "shows", "serves as a testament": "shows",
    "stands as a": "is a", "serves as a": "is a",
    "functions as a": "works as a", "operates as a": "works as a",
    "marks a significant": "is a big", "represents a significant": "is a big",
    "represents a shift": "is a change", "marking a": "showing a",
    "setting the stage for": "leading to", "key turning point": "turning point",
    "focal point": "main focus", "indelible mark": "lasting mark",
    "deeply rooted": "well established", "evolving landscape": "changing field",
    # Generic / promotional
    "driving sustainable growth": "growing steadily", "competitive advantage": "an edge",
    "drive growth": "grow", "drive innovation": "innovate",
    "unlock value": "add value", "maximize efficiency": "work more efficiently",
    "optimize performance": "improve performance", "enhance productivity": "improve productivity",
    "improve customer experience": "make things better for customers",
    "customer experiences": "customer experience", "gain a competitive edge": "get an edge",
    "stay ahead of the curve": "stay ahead", "stay competitive": "keep up",
    "best-in-class": "top", "world-class": "excellent", "industry-leading": "leading",
    "next-generation": "new", "future-proof": "long-lasting",
    "end-to-end solution": "full solution", "one-stop solution": "single solution",
    "value-added": "extra", "mission-critical": "essential",
    "core competencies": "core skills", "key takeaway": "main point",
    "actionable insights": "useful insights", "renowned": "well known",
    "groundbreaking": "new", "diverse array": "wide range",
    "natural beauty": "beauty", "nestled in": "located in",
    "in the heart of": "in the center of", "rich history": "long history",
    "rich culture": "strong culture",
}

# Sanity check: every word/phrase the detector can flag has a replacement.
_MISSING = [p for p in (*AI_ASSOCIATED_VOCAB, *GENERIC_PROMOTIONAL) if p not in VOCAB_REPLACEMENTS]
assert not _MISSING, f"rule_scrubber.VOCAB_REPLACEMENTS is missing entries for: {_MISSING}"

_VOCAB_PATTERN = re.compile(
    r"\b(?:" + "|".join(re.escape(p) for p in sorted(VOCAB_REPLACEMENTS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


def _preserve_case(original: str, replacement: str) -> str:
    return replacement[:1].upper() + replacement[1:] if original[:1].isupper() else replacement


def _substitute_vocab(text: str) -> str:
    def repl(m: "re.Match") -> str:
        matched = m.group(0)
        replacement = VOCAB_REPLACEMENTS.get(matched.lower())
        return _preserve_case(matched, replacement) if replacement else matched

    return _VOCAB_PATTERN.sub(repl, text)


def _strip_crutch_openers(text: str) -> str:
    """Removes a sentence-opener crutch ("Furthermore, ...") entirely when
    it's immediately followed by a comma — only that specific, unambiguous
    shape is touched, so this never risks cutting into a real sentence."""
    doc = get_nlp()(text)
    sentences = [sent.text for sent in doc.sents]
    fixed = []
    for sent in sentences:
        stripped = sent.strip()
        lower = stripped.lower()
        for crutch in SENTENCE_OPENER_CRUTCHES:
            prefix = crutch + ","
            if lower.startswith(prefix):
                rest = stripped[len(prefix):].lstrip()
                if rest:
                    rest = rest[:1].upper() + rest[1:]
                stripped = rest
                break
        if stripped:
            fixed.append(stripped)
    return " ".join(fixed)


def _reduce_em_dashes(text: str) -> str:
    """Matches the analyzer's own overuse threshold (score_em_dash_overuse):
    only acts once there are 2+ em dashes, otherwise leaves a single
    legitimate one alone."""
    if text.count("—") < 2:
        return text
    return re.sub(r"\s*—\s*", ", ", text)


def scrub_ai_signals(text: str) -> str:
    """Deterministically removes whichever of the detector's own flaggable
    surface patterns are safe to fix without risking meaning or grammar.
    Fails safe: returns the input unchanged if anything goes wrong."""
    try:
        result = _strip_crutch_openers(text)
        result = _substitute_vocab(result)
        result = _reduce_em_dashes(result)
        return result if result.strip() else text
    except Exception:
        return text
