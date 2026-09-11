"""
Curated phrase lists used by the rule-based analyzer.
These are intentionally editable — tune / extend them as you see false
positives or negatives in real usage.

Sourced from Wikipedia's "Signs of AI writing" community-maintained list
(https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing) plus phrases
independently and repeatedly documented as LLM "tells" by public write-ups
from AI-detection vendors (GPTZero, Originality.ai, Copyleaks, Turnitin,
Grammarly) and by large-scale collections of ChatGPT-era stock phrasing.
Real external detectors key on the same handful of surface families this
list targets — templated transitions, significance-inflation ("stands as a
testament"), hedge-free superlatives, and generic marketing filler — so
widening this list widens what both our own analyzer AND third-party
detectors will no longer find.
"""

# Words/phrases strongly associated with LLM-generated marketing/business prose.
AI_ASSOCIATED_VOCAB = [
    "revolutionizing", "revolutionize", "pivotal role", "pivotal",
    "transformative", "leverage", "leveraging", "streamline", "streamlining",
    "seamless", "seamlessly", "robust", "cutting-edge", "state-of-the-art",
    "in today's fast-paced world", "in today's digital age", "unlock the potential",
    "unlocking", "landscape", "ever-evolving", "dynamic landscape",
    "paradigm shift", "holistic", "synergy", "synergies", "delve into", "delve",
    "navigate the complexities", "navigating", "underscore", "underscores",
    "underscoring", "testament to", "it is important to note", "it's worth noting",
    "furthermore", "moreover", "in conclusion", "in summary", "harness the power",
    "harnessing", "elevate", "elevating", "empower", "empowering", "empowers",
    "foster", "fostering", "fosters", "bespoke", "tailored solutions",
    "game-changer", "game changing", "innovative solutions", "meticulously",
    "myriad", "plethora", "top-notch", "unparalleled", "unprecedented",
    # Wikipedia "Signs of AI writing" — core vocabulary
    "boasts", "boasts a", "bolstered", "crucial", "emphasizing", "enduring",
    "garner", "garnered", "intricate", "intricacies", "interplay", "tapestry",
    "vibrant", "align with", "aligns with", "resonates with", "enhance",
    "enhancing", "highlighting", "showcase", "showcasing", "showcases",
    "deep dive", "exemplifies", "profound", "commitment to",
    # Copula-avoidance / significance-inflation constructions
    "stands as a testament", "serves as a testament", "stands as a",
    "serves as a", "functions as a", "operates as a", "marks a significant",
    "represents a significant", "represents a shift", "marking a",
    "setting the stage for", "key turning point", "focal point",
    "indelible mark", "deeply rooted", "evolving landscape",
    # Additional stock transitions / hedge phrases widely flagged by
    # third-party AI detectors as templated LLM phrasing
    "when it comes to", "at the end of the day", "in this article",
    "in this piece", "let's dive in", "let's explore", "without further ado",
    "picture this", "imagine a world where", "in the world of",
    "plays a vital role", "play a vital role", "plays a crucial role",
    "play a crucial role", "is a great way to", "there are a few reasons why",
    "it's important to remember", "not to mention", "in essence",
    "in a nutshell", "all in all", "first and foremost", "needless to say",
    "the bottom line", "at its core", "as we navigate", "as we delve",
    "in the ever-changing world of", "in the realm of", "the realm of",
    "unlock new possibilities", "open up new possibilities",
    "take it to the next level", "push the boundaries", "redefine",
    "redefining", "reimagine", "reimagining", "trailblazing", "trailblazer",
    "a beacon of", "look no further", "look no further than",
    "whether you're a beginner or an expert", "in the digital era",
    "digital landscape", "fast-paced digital world", "ever-changing",
    "fast-paced", "in recent years", "in an era where",
    "as technology continues to evolve", "the possibilities are endless",
    "a double-edged sword", "food for thought", "a breath of fresh air",
    "unwavering commitment", "unwavering", "of utmost importance",
    "utmost importance", "gold standard", "tailor-made", "hallmark of",
    "cornerstone of", "vanguard", "at the forefront", "forefront of",
    "spearhead", "spearheading", "catalyst for", "linchpin", "keystone of",
    "no stone unturned", "in a world where", "it goes without saying",
    "the fact of the matter is", "suffice it to say",
    "when all is said and done", "sheds light on", "shed light on",
    "shine a light on", "stay ahead of the game", "unlock your potential",
    "take a deep dive", "deep-dive into", "a wealth of", "a myriad of",
    "a plethora of", "speaks volumes", "at the intersection of",
    "the intersection of",
]

# Generic / promotional filler common in AI-generated marketing copy
GENERIC_PROMOTIONAL = [
    "driving sustainable growth", "competitive advantage", "drive growth",
    "drive innovation", "unlock value", "maximize efficiency", "optimize performance",
    "enhance productivity", "improve customer experience", "customer experiences",
    "gain a competitive edge", "stay ahead of the curve", "stay competitive",
    "best-in-class", "world-class", "industry-leading", "next-generation",
    "future-proof", "end-to-end solution", "one-stop solution", "value-added",
    "mission-critical", "core competencies", "key takeaway", "actionable insights",
    # Wikipedia "Signs of AI writing" — promotional/descriptive buzzspeak
    "renowned", "groundbreaking", "diverse array", "natural beauty",
    "nestled in", "in the heart of", "rich history", "rich culture",
    # Additional promotional filler widely flagged by third-party detectors
    "unmatched quality", "unmatched", "exceptional quality", "peace of mind",
    "tailored to your needs", "one-of-a-kind", "second to none",
    "hassle-free", "user-friendly experience", "effortlessly",
    "elevate your", "to new heights", "unlock your", "unlock unparalleled",
    "designed to help you", "designed to empower", "empowering you to",
    "packed with features", "your one-stop shop", "your go-to",
    "game-changing", "revolutionary", "must-have", "a must for",
    "the ultimate guide", "everything you need to know",
    "in this comprehensive guide", "comprehensive guide", "dive deeper",
    "read on to discover", "read on to learn", "keep reading to",
    "stay tuned for",
]

# Common LLM sentence-opener crutches — repeated use is a signal. Only the
# phrase itself matters here (not a replacement), so this list is free to be
# broader than AI_ASSOCIATED_VOCAB / GENERIC_PROMOTIONAL.
SENTENCE_OPENER_CRUTCHES = [
    "furthermore", "moreover", "additionally", "in addition", "however",
    "therefore", "consequently", "notably", "importantly", "ultimately",
    "overall", "in conclusion", "in summary", "as a result",
    "in fact", "indeed", "crucially", "significantly", "essentially",
    "basically", "simply put", "put simply", "all things considered",
    "at the same time", "on the other hand", "similarly", "likewise",
    "conversely", "meanwhile", "nonetheless", "nevertheless", "thus",
    "hence", "accordingly", "specifically", "interestingly", "remarkably",
    "clearly", "undoubtedly", "unsurprisingly", "not surprisingly",
    "in other words", "that said", "with that said", "to that end",
    "in turn", "as such", "given this", "given that", "for this reason",
]
