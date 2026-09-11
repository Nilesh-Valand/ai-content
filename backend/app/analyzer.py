"""
Rule-based AI-writing likelihood analyzer.

No LLM calls happen in this module. Every signal is computed
deterministically from the text itself, so results are reproducible
and fully explainable.
"""
import re
from collections import Counter
from statistics import mean, pstdev
from typing import List

import numpy as np
import spacy
from lexicalrichness import LexicalRichness

from .models import DetectedPattern, SentenceScore
from .wordlists import (
    AI_ASSOCIATED_VOCAB,
    GENERIC_PROMOTIONAL,
    SENTENCE_OPENER_CRUTCHES,
)

_nlp = None


def get_nlp():
    """Lazy-load spaCy model once per process."""
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


# Signal weights — must sum to 1.0. Tune these based on observed accuracy.
#
# sentence_variation raised from 0.08 to 0.15 (external review, confirmed
# against this app's own output): it's the coefficient of variation of
# sentence word-counts — literal length burstiness, not structural variety
# — which published detector research (GLTR/DetectGPT-adjacent work) cites
# as one of the most consistently useful signals there is. At 0.08 it was
# underpowered specifically for MODERATE burstiness problems: the
# closed-loop's hard-cap correction trigger (_SIGNAL_HARD_CAP in
# humanizer.py) checks each signal's raw score independently, so a bad
# score still forces a revision regardless of blend weight — but a
# signal sitting just under that cap only moved the headline percentage a
# couple of points, letting it slide in the score that actually gates
# whether the closed loop stops early. Funded by trimming em_dash_overuse
# (0.04 -> 0.02: em dashes are now stripped unconditionally downstream in
# rule_scrubber regardless of score, so this signal has little marginal
# work left to do), repetition, diversity, and rule_of_three by 0.02/0.01
# each — all three are still real signals, just less centrally cited than
# burstiness and already well covered by the LLM's own instructions plus
# best-of-N candidate selection.
WEIGHTS = {
    "ai_vocab": 0.16,
    "predictability": 0.11,
    "repetition": 0.09,
    "diversity": 0.10,
    "generic_promo": 0.12,
    "rule_of_three": 0.05,
    "sentence_variation": 0.15,
    "specificity": 0.14,
    "negative_parallelism": 0.06,
    "em_dash_overuse": 0.02,
}

# "Not just X, but Y" / "it's not X, it's Y" — a well-documented LLM tic
# (Wikipedia calls it "negative parallelism"), rare enough in genuine human
# prose that even one occurrence in a short passage is meaningful.
NEGATIVE_PARALLELISM_PATTERNS = [
    re.compile(r"\bnot (?:just|only)\b[^.?!]{0,80}?\bbut\b", re.IGNORECASE),
    re.compile(
        r"\b(?:isn'?t|is not|it'?s not|wasn'?t|was not)\s+(?:just|only\s+)?"
        r"[^.?!—,;]{0,60}[,—;-]\s*(?:it'?s|it is|this is|that'?s|that is)\b",
        re.IGNORECASE,
    ),
]


def _find_phrase_hits(text_lower: str, phrases: List[str]) -> List[str]:
    hits = []
    for phrase in phrases:
        if phrase in text_lower:
            hits.append(phrase)
    return hits


def _ngrams(words: List[str], n: int) -> List[str]:
    return [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]


def score_ai_vocab(text_lower: str, word_count: int):
    hits = _find_phrase_hits(text_lower, AI_ASSOCIATED_VOCAB)
    # density: number of hits per 100 words, capped/normalized to 0-1
    density = (len(hits) / max(word_count, 1)) * 100
    score = min(density / 4.0, 1.0)  # 4+ hits per 100 words -> max score
    return score, hits


def score_generic_promo(text_lower: str, word_count: int):
    hits = _find_phrase_hits(text_lower, GENERIC_PROMOTIONAL)
    density = (len(hits) / max(word_count, 1)) * 100
    score = min(density / 3.0, 1.0)
    return score, hits


_RULE_OF_THREE_PATTERN = re.compile(
    r"([A-Za-z][\w\-]*(?:\s+[\w\-]+){0,4}),\s*"
    r"([A-Za-z][\w\-]*(?:\s+[\w\-]+){0,4}),?\s+(?:and|or)\s+"
    r"([A-Za-z][\w\-]*(?:\s+[\w\-]+){0,4})"
)

# Catches the same listicle cadence with NO coordinating conjunction at all
# -- "Apps can be lifesavers, calendars, task trackers, reminders." -- which
# the pattern above (built around "and"/"or") can't match. Real annotated
# output from a third-party detector flagged several sentences shaped
# exactly like this that the "and"-only pattern missed entirely. Each item
# capped at 3 words so an ordinary sentence with a comma or two doesn't
# trip it -- three-plus short parallel items in a row is specifically the
# tell being targeted here.
_ASYNDETIC_LIST_PATTERN = re.compile(
    r"\b(?:[A-Za-z][\w\-]*(?:\s+[\w\-]+){0,2},\s*){2,}"
    r"[A-Za-z][\w\-]*(?:\s+[\w\-]+){0,2}[.,;]"
)


def _spans_overlap(a: tuple, b: tuple) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def score_rule_of_three(text: str):
    """
    Detects patterns like:
      "enhancing productivity, fostering innovation, and driving growth"
    i.e. three comma/and-or-separated parallel phrases -- or the same
    cadence written with no conjunction at all (see
    _ASYNDETIC_LIST_PATTERN), which is just as much of a listicle-style
    tell but doesn't fit the "X, Y, and Z" shape.

    The two patterns' matches can overlap on the same underlying list (the
    asyndetic pattern's "two-or-more short items" can match a prefix of an
    "X, Y, and Z" construction the first pattern already caught) -- counted
    separately, that double-counts a single ordinary three-item list into
    a maxed-out score. Matches are tracked by span and deduplicated so the
    same stretch of text is never counted twice.
    """
    conjunction_iter = list(_RULE_OF_THREE_PATTERN.finditer(text))
    conjunction_spans = [m.span() for m in conjunction_iter]
    examples = [", ".join(m.groups()) for m in conjunction_iter]

    asyndetic_iter = [
        m for m in _ASYNDETIC_LIST_PATTERN.finditer(text)
        if not any(_spans_overlap(m.span(), s) for s in conjunction_spans)
    ]
    examples.extend(m.group(0).rstrip(" ,.;") for m in asyndetic_iter)

    total = len(conjunction_iter) + len(asyndetic_iter)
    # normalize: 2+ rule-of-three constructions in a short text is a strong signal
    score = min(total / 2.0, 1.0)
    return score, examples


def score_repetition(words: List[str]):
    if len(words) < 6:
        return 0.0, []
    bigrams = _ngrams(words, 2)
    trigrams = _ngrams(words, 3)
    combined = bigrams + trigrams
    counts = Counter(combined)
    repeated = [phrase for phrase, c in counts.items() if c > 1 and len(phrase.split()) >= 2]
    repeat_ratio = sum(c - 1 for c in counts.values() if c > 1) / max(len(combined), 1)
    score = min(repeat_ratio * 6.0, 1.0)
    return score, repeated[:8]


def score_diversity(text: str):
    """Lower lexical diversity -> higher AI likelihood."""
    words = re.findall(r"[A-Za-z']+", text)
    if len(words) < 10:
        return 0.0
    try:
        lex = LexicalRichness(text)
        mtld = lex.mtld(threshold=0.72)
        # Typical human MTLD ~60-100+, generic AI prose often lower (~40-70).
        # Normalize: MTLD <= 40 -> high AI score; MTLD >= 100 -> low AI score.
        normalized = 1.0 - min(max((mtld - 40) / 60.0, 0.0), 1.0)
        return round(normalized, 3)
    except Exception:
        ttr = len(set(w.lower() for w in words)) / len(words)
        return round(1.0 - min(ttr / 0.6, 1.0), 3)


def score_sentence_variation(sentences: List[str]):
    lengths = [len(s.split()) for s in sentences if s.strip()]
    if len(lengths) < 3:
        return 0.0
    m = mean(lengths)
    sd = pstdev(lengths)
    cv = sd / m if m > 0 else 0
    # low coefficient of variation (uniform sentence lengths) -> high AI likelihood
    score = 1.0 - min(cv / 0.5, 1.0)
    return round(max(score, 0.0), 3)


def score_predictability(sentences: List[str]):
    """
    Proxy for structural predictability: repeated sentence-opener crutches,
    repeated leading-word patterns, and consecutive sentences that share a
    parallel gerund-led template ("Saving a bit each week can build a fund.
    Reading a little each day can lead to finishing books. Practicing a
    skill can turn into real knowledge.") — a sentence-level version of
    "rule of three" that the comma-list regex in score_rule_of_three can't
    see, since each sentence here is a separate, grammatically complete
    one rather than a list within a single sentence.
    """
    if len(sentences) < 2:
        return 0.0, []
    openers = [s.strip().split()[0].lower().strip(",.") for s in sentences if s.strip()]
    opener_counts = Counter(openers)
    repeated_openers = [w for w, c in opener_counts.items() if c > 1]

    crutch_hits = []
    for s in sentences:
        s_lower = s.lower()
        for crutch in SENTENCE_OPENER_CRUTCHES:
            if s_lower.strip().startswith(crutch):
                crutch_hits.append(crutch)

    # Consecutive sentences that both open on a gerund ("-ing" word) are a
    # cheap, reliable proxy for that parallel-template pattern — two or
    # more in a row is the tell, using a DIFFERENT gerund each time (so
    # the exact-opener-word check above, which only catches the SAME
    # repeated word, doesn't already cover this).
    gerund_run_hits = []
    run_length = 0
    for w in openers:
        if w.endswith("ing") and len(w) > 4:
            run_length += 1
            if run_length >= 2:
                gerund_run_hits.append(w)
        else:
            run_length = 0

    signal = len(repeated_openers) + len(crutch_hits) + len(gerund_run_hits)
    score = min(signal / max(len(sentences) * 0.6, 1), 1.0)
    return round(score, 3), list(set(crutch_hits + gerund_run_hits))


def score_negative_parallelism(text: str):
    """Detects "not just X, but Y" / "it's not X, it's Y" style constructions."""
    matches = []
    for pattern in NEGATIVE_PARALLELISM_PATTERNS:
        matches.extend(m.group(0).strip() for m in pattern.finditer(text))
    # even a single occurrence in a short passage is a meaningful signal
    score = min(len(matches) * 0.6, 1.0)
    return round(score, 3), matches


def score_em_dash_overuse(text: str, sentence_count: int):
    """Overuse of em dashes ('—') is one of the most cited ChatGPT-era tells."""
    positions = [m.start() for m in re.finditer(r"—", text)]
    dash_count = len(positions)
    if dash_count < 2:
        return 0.0, dash_count, []

    density = dash_count / max(sentence_count, 1)
    score = min(density / 0.5, 1.0)  # ~1 em dash per 2 sentences -> max score

    snippets = []
    for pos in positions[:4]:
        start = max(0, pos - 20)
        end = min(len(text), pos + 21)
        snippets.append("…" + text[start:end].strip() + "…")

    return round(score, 3), dash_count, snippets


def score_specificity(doc, sentences: List[str]):
    """Lower presence of numbers/named entities/proper nouns -> higher AI likelihood."""
    if not sentences:
        return 0.0
    entity_count = len(doc.ents)
    number_count = len(re.findall(r"\b\d+([.,]\d+)?%?\b", doc.text))
    proper_noun_count = sum(1 for tok in doc if tok.pos_ == "PROPN")
    concrete_signals = entity_count + number_count + proper_noun_count
    density = concrete_signals / len(sentences)
    # 0 concrete signals per sentence -> max AI score; 1+ per sentence -> low
    score = 1.0 - min(density / 1.0, 1.0)
    return round(score, 3)


def severity_from_score(score: float) -> str:
    if score >= 0.6:
        return "High"
    if score >= 0.3:
        return "Medium"
    return "Low"


def analyze_text(text: str):
    nlp = get_nlp()
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    words = re.findall(r"[A-Za-z']+", text)
    text_lower = text.lower()
    word_count = len(words)

    ai_vocab_score, ai_vocab_hits = score_ai_vocab(text_lower, word_count)
    generic_score, generic_hits = score_generic_promo(text_lower, word_count)
    rule3_score, rule3_examples = score_rule_of_three(text)
    repetition_score, repetition_hits = score_repetition(words)
    diversity_score = score_diversity(text)
    variation_score = score_sentence_variation(sentences)
    predictability_score, predictability_hits = score_predictability(sentences)
    specificity_score = score_specificity(doc, sentences)
    neg_parallel_score, neg_parallel_hits = score_negative_parallelism(text)
    em_dash_score, em_dash_count, em_dash_snippets = score_em_dash_overuse(text, len(sentences))

    signals = {
        "ai_vocab": ai_vocab_score,
        "predictability": predictability_score,
        "repetition": repetition_score,
        "diversity": diversity_score,
        "generic_promo": generic_score,
        "rule_of_three": rule3_score,
        "sentence_variation": variation_score,
        "specificity": specificity_score,
        "negative_parallelism": neg_parallel_score,
        "em_dash_overuse": em_dash_score,
    }

    overall_raw = sum(signals[k] * WEIGHTS[k] for k in WEIGHTS)
    overall_pct = round(overall_raw * 100, 1)

    # Confidence: based on agreement (low spread) among the signals,
    # and how far the overall score sits from the uncertain midpoint.
    spread = pstdev(list(signals.values()))
    distance_from_mid = abs(overall_raw - 0.5)
    if spread < 0.2 and distance_from_mid > 0.2:
        confidence = "High"
    elif spread < 0.3:
        confidence = "Medium"
    else:
        confidence = "Low"

    detected_patterns = [
        DetectedPattern(
            pattern="AI-associated vocabulary",
            severity=severity_from_score(ai_vocab_score),
            score=round(ai_vocab_score, 3),
            examples=ai_vocab_hits[:6],
        ),
        DetectedPattern(
            pattern="Generic / promotional language",
            severity=severity_from_score(generic_score),
            score=round(generic_score, 3),
            examples=generic_hits[:6],
        ),
        DetectedPattern(
            pattern="Rule-of-three structure",
            severity=severity_from_score(rule3_score),
            score=round(rule3_score, 3),
            examples=rule3_examples[:4],
        ),
        DetectedPattern(
            pattern="Repetition",
            severity=severity_from_score(repetition_score),
            score=round(repetition_score, 3),
            examples=repetition_hits,
        ),
        DetectedPattern(
            pattern="Low vocabulary diversity",
            severity=severity_from_score(diversity_score),
            score=round(diversity_score, 3),
            examples=[],
        ),
        DetectedPattern(
            pattern="Low sentence-length variation",
            severity=severity_from_score(variation_score),
            score=round(variation_score, 3),
            examples=[],
        ),
        DetectedPattern(
            pattern="Predictable sentence structure",
            severity=severity_from_score(predictability_score),
            score=round(predictability_score, 3),
            examples=predictability_hits[:6],
        ),
        DetectedPattern(
            pattern="Lack of specificity",
            severity=severity_from_score(specificity_score),
            score=round(specificity_score, 3),
            examples=[],
        ),
        DetectedPattern(
            pattern='Negative parallelism ("not just X, but Y")',
            severity=severity_from_score(neg_parallel_score),
            score=round(neg_parallel_score, 3),
            examples=neg_parallel_hits[:4],
        ),
        DetectedPattern(
            pattern="Em dash overuse",
            severity=severity_from_score(em_dash_score),
            score=round(em_dash_score, 3),
            examples=em_dash_snippets,
        ),
    ]

    # Sentence-level scoring using a lighter-weight blend of applicable signals
    sentence_scores = []
    for i, sent in enumerate(sentences):
        s_lower = sent.lower()
        s_words = re.findall(r"[A-Za-z']+", sent)
        s_word_count = max(len(s_words), 1)

        s_vocab_hits = _find_phrase_hits(s_lower, AI_ASSOCIATED_VOCAB)
        s_generic_hits = _find_phrase_hits(s_lower, GENERIC_PROMOTIONAL)
        s_vocab_density = min((len(s_vocab_hits) / s_word_count) * 100 / 4.0, 1.0)
        s_generic_density = min((len(s_generic_hits) / s_word_count) * 100 / 3.0, 1.0)

        sent_doc = nlp(sent)
        s_concrete = len(sent_doc.ents) + len(re.findall(r"\b\d+([.,]\d+)?%?\b", sent)) + \
            sum(1 for tok in sent_doc if tok.pos_ == "PROPN")
        s_specificity = 1.0 - min(s_concrete / 1.0, 1.0)
        s_neg_parallel = any(p.search(sent) for p in NEGATIVE_PARALLELISM_PATTERNS)
        s_rule3_score, _s_rule3_examples = score_rule_of_three(sent)

        blended = (
            0.30 * s_vocab_density +
            0.20 * s_generic_density +
            0.20 * s_specificity +
            0.15 * s_rule3_score +
            (0.15 if s_neg_parallel else 0.0)
        )
        sentence_scores.append(
            SentenceScore(index=i + 1, text=sent, ai_likelihood=round(blended * 100, 1))
        )

    highlighted_phrases = list(set(ai_vocab_hits + generic_hits + rule3_examples + neg_parallel_hits))

    return {
        "overall_pct": overall_pct,
        "confidence": confidence,
        "detected_patterns": detected_patterns,
        "sentence_scores": sentence_scores,
        "highlighted_phrases": highlighted_phrases,
        "signals": signals,
    }
