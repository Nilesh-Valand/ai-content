"""
Local perplexity/burstiness scorer — a lightweight, self-hosted proxy for
the statistical signal that real third-party AI detectors are documented
to actually use.

Why this exists: analyzer.py is a surface-pattern matcher (buzzwords,
sentence-opener crutches, em dash counts, ...). It's fast and fully
explainable, but structurally blind to the thing ZeroGPT's own published
methodology says it primarily measures — per-token predictability under a
language model ("perplexity"), and how much that predictability varies
sentence-to-sentence across a document ("burstiness"). See:
  https://gptzero.me/news/perplexity-and-burstiness-what-is-it/
  https://support.gptzero.me/articles/9585228410-how-do-i-interpret-burstiness-or-perplexity
Text can use none of analyzer.py's flagged vocabulary and still read as
consistently, mechanically "smooth" to a language model — which is exactly
what a heavily-paraphrased LLM rewrite tends to do. This module gives the
humanizer's closed-loop refinement (humanizer.py) a second, independent
signal that targets that blind spot directly, using a small local causal
LM (GPT-2) rather than this app's own hand-written rules.

Calibration note: the absolute perplexity numbers here are specific to
GPT-2-small's own vocabulary and scale, not directly comparable to
whatever (larger, differently-trained) model a given third-party detector
uses internally — there's no universal "85 means human" constant that
transfers across models. The thresholds below were set empirically against
sample AI-generic vs. humanized document pairs (see the module's dev notes)
rather than copied from any single vendor's published number, and this is
meant as one directional signal among several, not a precise replica of
any specific detector.

Loads lazily and fails soft: if the model can't be loaded for any reason
(dependency missing, no network for the first download, etc.), scoring
returns None and callers fall back to the rule-based analyzer alone.
"""
import math
import statistics
from typing import List, Optional, Tuple

_tokenizer = None
_model = None
_load_failed = False

# A sentence shorter than this doesn't carry enough signal for a
# per-sentence perplexity to mean much (and can swing wildly on a couple of
# tokens), so it's excluded from both the mean and the burstiness spread.
_MIN_SENTENCE_WORDS = 4

# GPT-2-small mean perplexity below this reads as strongly AI-like
# (predictable, "safest next word" phrasing); above this reads as strongly
# human-like. Empirically, generic AI-buzzword prose commonly lands
# roughly in the 10-40 range on this model/tokenizer, while genuinely
# varied human or well-humanized prose commonly lands 60-120+.
_PPL_AI_FLOOR = 15.0
_PPL_HUMAN_CEILING = 60.0

# Coefficient of variation (stdev/mean) of per-sentence perplexity across
# the document. Below this reads as uniformly predictable throughout
# (AI-like); above this reads as bursty/varied (human-like).
_BURST_AI_FLOOR = 0.35
_BURST_HUMAN_CEILING = 0.75


def _get_model():
    global _tokenizer, _model, _load_failed
    if _load_failed:
        return None, None
    if _model is None:
        try:
            from transformers import GPT2LMHeadModel, GPT2TokenizerFast
            _tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
            _model = GPT2LMHeadModel.from_pretrained("gpt2")
            _model.eval()
        except Exception:
            _load_failed = True
            return None, None
    return _tokenizer, _model


def _sentence_perplexity(sentence: str) -> Optional[float]:
    tokenizer, model = _get_model()
    if tokenizer is None:
        return None
    try:
        import torch
        encodings = tokenizer(sentence, return_tensors="pt")
        input_ids = encodings.input_ids
        if input_ids.shape[1] < 2:
            return None
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
        return math.exp(outputs.loss.item())
    except Exception:
        return None


def _normalize(value: float, floor: float, ceiling: float) -> float:
    """0.0 at/above ceiling, 1.0 at/below floor, linear between."""
    if ceiling <= floor:
        return 0.0
    return 1.0 - min(max((value - floor) / (ceiling - floor), 0.0), 1.0)


def score_document(sentences: List[str]) -> Optional[Tuple[float, float, float]]:
    """Scores a full document's sentences (already split by the caller —
    see analyzer.get_nlp()). Returns (mean_perplexity, burstiness_cv,
    ai_likelihood) where ai_likelihood is 0.0 (reads human-like to GPT-2)
    to 1.0 (reads AI-like), or None if the model isn't available or there
    isn't enough usable text (fewer than 2 sentences of at least
    _MIN_SENTENCE_WORDS words each) to compute a meaningful document-level
    signal — single-sentence perplexity alone is too noisy to trust; this
    is deliberately a document-level-only signal."""
    usable = [s for s in sentences if len(s.split()) >= _MIN_SENTENCE_WORDS]
    ppls = [p for p in (_sentence_perplexity(s) for s in usable) if p is not None]
    if len(ppls) < 2:
        return None

    mean_ppl = statistics.mean(ppls)
    stdev = statistics.pstdev(ppls)
    burstiness = stdev / mean_ppl if mean_ppl > 0 else 0.0

    ppl_score = _normalize(mean_ppl, _PPL_AI_FLOOR, _PPL_HUMAN_CEILING)
    burst_score = _normalize(burstiness, _BURST_AI_FLOOR, _BURST_HUMAN_CEILING)

    # Mean perplexity carries more weight than burstiness: in practice it's
    # the more stable of the two on short documents (a handful of
    # sentences isn't much to estimate a variance from), so it should
    # dominate the blend rather than be diluted by a noisier co-signal.
    ai_likelihood = round(0.7 * ppl_score + 0.3 * burst_score, 3)
    return round(mean_ppl, 2), round(burstiness, 3), ai_likelihood
