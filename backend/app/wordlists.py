"""
Curated phrase lists used by the rule-based analyzer.
These are intentionally editable — tune / extend them as you see false
positives or negatives in real usage.
"""

# Words/phrases strongly associated with LLM-generated marketing/business prose.
# Extended per Wikipedia's "Signs of AI writing" community-maintained list:
# https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing
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
]

# Common LLM sentence-opener crutches — repeated use is a signal
SENTENCE_OPENER_CRUTCHES = [
    "furthermore", "moreover", "additionally", "in addition", "however",
    "therefore", "consequently", "notably", "importantly", "ultimately",
    "overall", "in conclusion", "in summary", "as a result",
]
