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
WEIGHTS = {
    "ai_vocab": 0.18,
    "predictability": 0.12,
    "repetition": 0.12,
    "diversity": 0.13,
    "generic_promo": 0.13,
    "rule_of_three": 0.08,
    "sentence_variation": 0.09,
    "specificity": 0.15,
}


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


def score_rule_of_three(text: str):
    """
    Detects patterns like:
      "enhancing productivity, fostering innovation, and driving growth"
    i.e. three comma/and-separated parallel phrases.
    """
    pattern = re.compile(
        r"([A-Za-z][\w\-]*(?:\s+[\w\-]+){0,4}),\s*"
        r"([A-Za-z][\w\-]*(?:\s+[\w\-]+){0,4}),?\s+and\s+"
        r"([A-Za-z][\w\-]*(?:\s+[\w\-]+){0,4})"
    )
    matches = pattern.findall(text)
    examples = [", ".join(m) for m in matches]
    # normalize: 2+ rule-of-three constructions in a short text is a strong signal
    score = min(len(matches) / 2.0, 1.0)
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
    Proxy for structural predictability: repeated sentence-opener crutches
    and repeated leading-word patterns across sentences.
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

    signal = len(repeated_openers) + len(crutch_hits)
    score = min(signal / max(len(sentences) * 0.6, 1), 1.0)
    return round(score, 3), list(set(crutch_hits))


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

    signals = {
        "ai_vocab": ai_vocab_score,
        "predictability": predictability_score,
        "repetition": repetition_score,
        "diversity": diversity_score,
        "generic_promo": generic_score,
        "rule_of_three": rule3_score,
        "sentence_variation": variation_score,
        "specificity": specificity_score,
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

        blended = (
            0.40 * s_vocab_density +
            0.30 * s_generic_density +
            0.30 * s_specificity
        )
        sentence_scores.append(
            SentenceScore(index=i + 1, text=sent, ai_likelihood=round(blended * 100, 1))
        )

    highlighted_phrases = list(set(ai_vocab_hits + generic_hits + rule3_examples))

    return {
        "overall_pct": overall_pct,
        "confidence": confidence,
        "detected_patterns": detected_patterns,
        "sentence_scores": sentence_scores,
        "highlighted_phrases": highlighted_phrases,
        "signals": signals,
    }
