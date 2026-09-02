# AI Content Detector

A tool that scores text for "AI-writing likelihood" using **rule-based
signal detection only** — no LLM in the scoring path, so results are
deterministic and explainable. An LLM (via Groq) is used **only** to phrase
natural-language rewrite suggestions after scoring is complete.

```
frontend/   Next.js 14 (App Router, TypeScript, Tailwind) — editorial "marked
            manuscript" UI
backend/    FastAPI — rule-based analyzer (spaCy + wordlists + stats) and a
            Groq-powered /suggestions step
```

## How scoring works

Eight signals are computed deterministically per `backend/app/analyzer.py`,
each normalized to 0–1, then combined with tunable weights:

| Signal | Method |
|---|---|
| AI-associated vocabulary | wordlist density (`app/wordlists.py`) |
| Generic / promotional language | wordlist density |
| Rule-of-three structure | regex for parallel `X, Y, and Z` constructions |
| Repetition | repeated bigrams/trigrams |
| Vocabulary diversity | MTLD via `lexicalrichness` (falls back to TTR) |
| Sentence-length variation | coefficient of variation of sentence lengths |
| Predictability | repeated sentence openers / transition-word crutches |
| Specificity | density of named entities, numbers, proper nouns (spaCy) |

Confidence is derived from how much the eight signals agree with each
other (low spread + a score far from 50% → High confidence).

Weights live at the top of `analyzer.py` as `WEIGHTS` — tune them against
your own labeled examples if you want to improve accuracy.

## Backend setup

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
cp .env.example .env   # fill in GROQ_API_KEY, check GROQ_MODEL is a currently
                        # available model in your Groq console
uvicorn app.main:app --reload --port 8000
```

`GROQ_API_KEY` is only required for the suggestions step — if it's missing,
`/analyze` still returns full scoring, patterns, and sentence breakdown; the
`suggestions` array just comes back empty.

## Frontend setup

```bash
cd frontend
npm install
cp .env.local.example .env.local   # points at http://localhost:8000 by default
npm run dev
```

Open http://localhost:3000, paste text, click **Analyze Content**.

## API

`POST /analyze`
```json
{ "content": "your text here", "include_suggestions": true }
```
Returns `overall`, `detected_patterns`, `sentence_scores`,
`highlighted_phrases`, and `suggestions` — see `backend/app/models.py` for
the full schema.

## Notes

- Groq model names change over time — check the Groq console for currently
  available models before deploying; `GROQ_MODEL` in `.env` defaults to a
  placeholder you should verify.
- All wordlists in `app/wordlists.py` are a starting point, not exhaustive —
  expect to tune them (and the signal weights) against real examples.
