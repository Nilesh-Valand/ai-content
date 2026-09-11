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
  - Emoji of every kind -> removed unconditionally
  - Collapsed paragraph structure -> paragraph breaks re-inserted to match
    the original's structure, proportioned by paragraph length (see
    restore_paragraph_structure) — a wall of text with no breaks is itself
    a recognizable AI tell, and an LLM revision pass can quietly flatten
    structure even when told not to, so this is enforced mechanically
    rather than left to the prompt alone.

Signals that are NOT touched here on purpose, because a safe rule-based fix
doesn't exist for them: rule-of-three restructuring, sentence-length
variation, lexical diversity, repeated-bigram removal, negative parallelism,
and — most importantly — "lack of specificity", which would require
inventing facts that were never in the original text.

Fails safe: any unexpected error here returns the input text unchanged
rather than risking mangled output.
"""
import re
from typing import Dict, List, Optional

from .analyzer import get_nlp
from .wordlists import AI_ASSOCIATED_VOCAB, GENERIC_PROMOTIONAL, SENTENCE_OPENER_CRUTCHES

try:
    # Comprehensive, actively-maintained Unicode emoji data (covers keycap
    # sequences like "1️⃣", ZWJ sequences, skin-tone modifiers, and
    # blocks like Miscellaneous Technical U+2300-23FF that a hand-rolled
    # range list is easy to miss — e.g. U+23F1 STOPWATCH). Falls back to the
    # hand-rolled ranges below if the package isn't installed.
    import emoji as _emoji_lib
except ImportError:
    _emoji_lib = None

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
    "myriad": "many", "myriad of": "many",
    "plethora": "a lot of", "plethora of": "a lot of", "top-notch": "great",
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
    # Additional stock transitions / hedge phrases
    "when it comes to": "with", "at the end of the day": "in the end",
    "in this article": "here", "in this piece": "here",
    "let's dive in": "let's get started", "let's explore": "let's look at",
    "without further ado": "so", "picture this": "consider this",
    "imagine a world where": "consider a case where", "in the world of": "in",
    "plays a vital role": "matters", "play a vital role": "matter",
    "plays a crucial role": "matters", "play a crucial role": "matter",
    "is a great way to": "helps", "there are a few reasons why": "here's why",
    "it's important to remember": "remember", "not to mention": "and",
    "in essence": "in short", "in a nutshell": "in short",
    "all in all": "overall", "first and foremost": "first",
    "needless to say": "of course", "the bottom line": "the key point",
    "at its core": "at heart", "as we navigate": "as we deal with",
    "as we delve": "as we look into",
    "in the ever-changing world of": "in", "in the realm of": "in",
    "the realm of": "the field of",
    "unlock new possibilities": "open up new options",
    "open up new possibilities": "open up new options",
    "take it to the next level": "improve it further",
    "push the boundaries": "go further", "redefine": "change",
    "redefining": "changing", "reimagine": "rethink",
    "reimagining": "rethinking", "trailblazing": "pioneering",
    "trailblazer": "pioneer", "a beacon of": "a symbol of",
    "look no further": "consider this", "look no further than": "consider",
    "whether you're a beginner or an expert": "no matter your experience level",
    "in the digital era": "today", "digital landscape": "digital space",
    "fast-paced digital world": "fast-moving digital space",
    "ever-changing": "changing", "fast-paced": "fast-moving",
    "in recent years": "recently", "in an era where": "at a time when",
    "as technology continues to evolve": "as technology changes",
    "the possibilities are endless": "there's a lot you can do",
    "a double-edged sword": "a trade-off",
    "food for thought": "something to consider",
    "a breath of fresh air": "a welcome change",
    "unwavering commitment": "strong commitment", "unwavering": "steady",
    "of utmost importance": "very important", "utmost importance": "high importance",
    "gold standard": "benchmark", "tailor-made": "custom",
    "hallmark of": "sign of", "cornerstone of": "basis of",
    "vanguard": "front", "at the forefront of": "leading",
    "at the forefront": "leading",
    "forefront of": "front of", "spearhead": "lead",
    "spearheading": "leading", "catalyst for": "driver of",
    "linchpin": "key part", "keystone of": "key part of",
    "no stone unturned": "covered everything",
    "in a world where": "at a time when",
    "it goes without saying": "clearly",
    "the fact of the matter is": "the truth is",
    "suffice it to say": "put simply",
    "when all is said and done": "in the end",
    "sheds light on": "explains", "shed light on": "explain",
    "shine a light on": "points out",
    "stay ahead of the game": "stay ahead",
    "unlock your potential": "reach your potential",
    "take a deep dive": "look closely", "deep-dive into": "look closely at",
    "a wealth of": "a lot of", "a myriad of": "many",
    "a plethora of": "a lot of", "speaks volumes": "says a lot",
    "at the intersection of": "combining", "the intersection of": "the overlap of",
    # Additional promotional filler
    "unmatched quality": "high quality", "unmatched": "excellent",
    "exceptional quality": "high quality", "peace of mind": "confidence",
    "tailored to your needs": "customized for you", "one-of-a-kind": "unique",
    "second to none": "the best", "hassle-free": "easy",
    "user-friendly experience": "easy-to-use experience",
    "effortlessly": "easily", "elevate your": "improve your",
    "to new heights": "further", "unlock your": "improve your",
    "unlock unparalleled": "provide excellent",
    "designed to help you": "built to help you",
    "designed to empower": "built to help",
    "empowering you to": "helping you",
    "packed with features": "full of features",
    "your one-stop shop": "your single source", "your go-to": "your top choice",
    "game-changing": "significant", "revolutionary": "new",
    "must-have": "essential", "a must for": "essential for",
    "the ultimate guide": "a complete guide",
    "everything you need to know": "what you need to know",
    "in this comprehensive guide": "in this guide",
    "comprehensive guide": "full guide", "dive deeper": "look closer",
    "read on to discover": "keep reading for",
    "read on to learn": "keep reading for",
    "keep reading to": "continue reading to",
    "stay tuned for": "watch for",
    # Canonical short-maxim clichés
    "is key to": "matters for", "is key": "matters",
    "practice makes perfect": "practice helps",
    "slow and steady wins the race": "steady effort pays off",
    "what matters most": "the main thing",
    "what matters is": "the main thing is",
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


# A line that IS a list item: a real bullet, a markdown-style hyphen/asterisk
# bullet (the LLM's own inconsistent default — see _normalize_list_markers),
# or a "1. "/"1)" numbered marker, each followed by real content.
_LIST_LINE_PATTERN = re.compile(r"^\s*(?:[•‣◦⁃\-*]|\d+[.)])\s+\S")


def _is_list_line(line: str) -> bool:
    return bool(_LIST_LINE_PATTERN.match(line))


def _strip_crutch_from_sentence(sentence: str) -> str:
    stripped = sentence.strip()
    lower = stripped.lower()
    for crutch in SENTENCE_OPENER_CRUTCHES:
        prefix = crutch + ","
        if lower.startswith(prefix):
            rest = stripped[len(prefix):].lstrip()
            if rest:
                rest = rest[:1].upper() + rest[1:]
            return rest
    return stripped


def _strip_crutch_openers(text: str) -> str:
    """Removes a sentence-opener crutch ("Furthermore, ...") entirely when
    it's immediately followed by a comma — only that specific, unambiguous
    shape is touched, so this never risks cutting into a real sentence.

    Processes paragraph by paragraph (rather than running spaCy sentence
    splitting across the whole document and rejoining everything with a
    single space) so paragraph breaks survive this pass intact. Without
    that, every call would flatten the entire document into one block and
    leave restore_paragraph_structure (see scrub_ai_signals) to
    reconstruct paragraphs from scratch by word-weight redistribution —
    which reliably gets the paragraph *count* right but doesn't preserve
    which sentences the model actually grouped together, since it's
    working from a proportional guess rather than the model's real
    grouping. Splitting per-paragraph here means that reconstruction step
    only has to do real work on genuinely collapsed input, which is what
    it's actually for.

    A "paragraph" that's actually a list block (its lines are bullet/number
    items) is handled per-line instead: each line is crutch-stripped on its
    own and rejoined with "\\n", never merged into one space-joined run —
    the ordinary prose path's sentence-split-then-rejoin-with-" " would
    otherwise collapse "• Save time... • Smarter choices..." into a single
    run-on line, destroying the list."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    fixed_paragraphs = []
    for para in paragraphs:
        lines = para.split("\n")
        if any(_is_list_line(line) for line in lines):
            fixed_lines = [_strip_crutch_from_sentence(line) for line in lines if line.strip()]
            fixed_lines = [line for line in fixed_lines if line]
            if fixed_lines:
                fixed_paragraphs.append("\n".join(fixed_lines))
            continue

        doc = get_nlp()(para)
        sentences = [sent.text for sent in doc.sents]
        fixed = [f for f in (_strip_crutch_from_sentence(s) for s in sentences) if f]
        if fixed:
            fixed_paragraphs.append(" ".join(fixed))
    return "\n\n".join(fixed_paragraphs)


# Markdown-style bullet markers (hyphen, asterisk) the LLM defaults to
# despite being told to use a real bullet character — normalized here
# deterministically so the output is never at the mercy of the model
# actually following that instruction. Numbered markers ("1. ") are left
# alone since they're already what we want.
_LIST_MARKER_PATTERN = re.compile(r"^(\s*)[\-*]\s+(?=\S)", re.MULTILINE)


def _normalize_list_markers(text: str) -> str:
    return _LIST_MARKER_PATTERN.sub(lambda m: m.group(1) + "• ", text)


def _reduce_em_dashes(text: str) -> str:
    """Removes every em dash, not just this app's own overuse threshold
    (2+, see score_em_dash_overuse) — third-party detectors and AI-savvy
    readers commonly treat even a single em dash as a tell, since it's a
    punctuation mark ChatGPT-era models reach for far more than most human
    writers do. A comma reads naturally in virtually every place an em dash
    would have gone."""
    if "—" not in text:
        return text
    return re.sub(r"\s*—\s*", ", ", text)


# Fallback only — used if the `emoji` package (see import at top) isn't
# installed. Emoji + pictograph/symbol ranges (not spoken-language
# punctuation), plus the variation-selector, zero-width-joiner, and
# combining-enclosing-keycap code points used to combine them into sequences
# like "1️⃣" or "👩‍💻". The user's writing profile should never come back
# with decoration the LLM added on its own — this is a hard, unconditional
# strip, independent of whatever the prompt asked for.
_EMOJI_PATTERN = re.compile(
    "["
    "\U0001F300-\U0001FAFF"  # symbols & pictographs, transport, supplemental symbols, emoji extensions
    "\U00002600-\U000027BF"  # misc symbols, dingbats (includes ☀ ✅ ✨ ➡ etc.)
    "\U0001F1E6-\U0001F1FF"  # regional indicator letters (flag emoji)
    "\U00002190-\U000021FF"  # arrows (➡ often rendered from this block too)
    "\U00002300-\U000023FF"  # misc technical (⏱ ⌛ ⏰ ⏳ ⌚ etc.)
    "\U00002B00-\U00002BFF"  # misc symbols & arrows (⭐ ⬆ ⬇ etc.)
    "\U0000FE0F"             # variation selector-16 (emoji presentation)
    "\U0000200D"             # zero-width joiner (combines emoji sequences)
    "\U000020E3"             # combining enclosing keycap (1️⃣ 2️⃣ ... #️⃣)
    "]+",
)


def _strip_emoji(text: str) -> str:
    if _emoji_lib is not None:
        return _emoji_lib.replace_emoji(text, replace="")
    return _EMOJI_PATTERN.sub("", text)


# Invisible/lookalike unicode characters an LLM occasionally emits (a
# non-breaking hyphen instead of a plain "-", a non-breaking space, a
# zero-width space, a stray byte-order mark) -- harmless to detectors, but
# copy-pasted "perfect" output shouldn't carry invisible characters that can
# render oddly or break search/highlighting in whatever tool it lands in.
# Written as \u escapes rather than the literal glyphs, since most of
# these are by definition invisible in a normal editor view.
_TYPOGRAPHY_NORMALIZATION = {
    "\u2011": "-",  # non-breaking hyphen
    "\u00a0": " ",  # non-breaking space
    "\u200b": "",   # zero-width space
    "\ufeff": "",   # byte-order mark
}
_TYPOGRAPHY_PATTERN = re.compile("|".join(_TYPOGRAPHY_NORMALIZATION))


def _normalize_typography(text: str) -> str:
    return _TYPOGRAPHY_PATTERN.sub(lambda m: _TYPOGRAPHY_NORMALIZATION[m.group(0)], text)


def _normalize_whitespace(text: str) -> str:
    """Cleanup pass after substitutions/removals: collapses doubled spaces
    and fixes stray space-before-punctuation that a phrase swap or an
    emoji strip can leave behind, without touching intentional formatting
    like paragraph breaks."""
    lines = text.split("\n")
    fixed_lines = []
    for line in lines:
        fixed = re.sub(r"[ \t]{2,}", " ", line)
        fixed = re.sub(r"\s+([,.!?;:])", r"\1", fixed)
        fixed_lines.append(fixed.strip())
    return "\n".join(fixed_lines)


def _split_sentences(text: str) -> List[str]:
    """Sentence-splits `text`, collapsing each sentence's internal
    whitespace to single spaces. Needed because spaCy's sentencizer doesn't
    always break exactly at a pre-existing blank-line paragraph boundary —
    e.g. a heading line with no terminal punctuation, followed by a blank
    line and then body text, commonly comes back as one "sentence" spanning
    both. Splitting on blank lines first (before collapsing whitespace)
    catches that: without it, two chunks that were really on opposite sides
    of a paragraph break would get silently concatenated with nothing but a
    space between them once whitespace is collapsed, or a literal embedded
    "\\n\\n" would ride along into whichever caller re-joins these with
    spaces (restore_paragraph_structure), producing an extra paragraph break
    the caller never asked for."""
    doc = get_nlp()(text)
    pieces: List[str] = []
    for sent in doc.sents:
        raw = sent.text.strip()
        if not raw:
            continue
        for chunk in re.split(r"\n\s*\n", raw):
            chunk = re.sub(r"\s+", " ", chunk).strip()
            if chunk:
                pieces.append(chunk)
    return pieces


def _split_paragraphs(text: str) -> List[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


# Below this fraction of the original paragraph count, `text` is treated as
# having flattened the source's structure and gets paragraph breaks
# mechanically restored (see restore_paragraph_structure). Mirrors
# humanizer._MIN_PARAGRAPH_RATIO — kept as a separate constant since this
# module has no dependency on humanizer.py.
_MIN_PARAGRAPH_RATIO = 0.7


def restore_paragraph_structure(original: str, text: str) -> str:
    """If `text` has collapsed the paragraph breaks `original` had (e.g. an
    LLM revision pass merged several paragraphs into one dense block, even
    after being told not to), deterministically re-inserts paragraph breaks
    into `text` at sentence boundaries — proportioned to match each of
    `original`'s paragraphs' relative length, so a long original paragraph
    still maps to a long stretch of the rewrite and a short one to a short
    stretch. This only ever inserts blank lines between existing sentences;
    it never touches wording, so unlike an LLM retry it can't introduce a
    meaning or grammar defect while fixing this.

    A no-op if the original doesn't have enough paragraphs to make this
    meaningful, if `text`'s paragraph count already looks fine, or if `text`
    doesn't have enough sentences to redistribute across the target count.
    Fails safe: returns `text` unchanged if anything goes wrong.
    """
    try:
        orig_paras = _split_paragraphs(original)
        n = len(orig_paras)
        if n < 3:
            return text

        if len(_split_paragraphs(text)) >= max(2, round(n * _MIN_PARAGRAPH_RATIO)):
            return text

        sentences = _split_sentences(text)
        total_sentences = len(sentences)
        if total_sentences < n:
            return text

        weights = [max(len(p.split()), 1) for p in orig_paras]
        total_weight = sum(weights)

        groups: List[List[str]] = []
        start = 0
        cum_weight = 0
        for i, w in enumerate(weights[:-1]):
            cum_weight += w
            remaining_after = n - len(groups) - 1
            lo = start + 1
            hi = total_sentences - remaining_after
            boundary = max(lo, min(round(total_sentences * cum_weight / total_weight), hi))
            groups.append(sentences[start:boundary])
            start = boundary
        groups.append(sentences[start:])

        return "\n\n".join(" ".join(g) for g in groups if g)
    except Exception:
        return text


def scrub_ai_signals(text: str, original: Optional[str] = None) -> str:
    """Deterministically removes whichever of the detector's own flaggable
    surface patterns are safe to fix without risking meaning or grammar.
    Pass `original` (the pre-humanize source text) to also mechanically
    restore paragraph structure if it got collapsed — omit it where no
    original is available/relevant.
    Fails safe: returns the input unchanged if anything goes wrong."""
    try:
        result = _strip_crutch_openers(text)
        result = _normalize_list_markers(result)
        result = _substitute_vocab(result)
        result = _reduce_em_dashes(result)
        result = _strip_emoji(result)
        result = _normalize_typography(result)
        result = _normalize_whitespace(result)
        if original:
            result = restore_paragraph_structure(original, result)
        return result if result.strip() else text
    except Exception:
        return text
