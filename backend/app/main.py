import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

from .analyzer import analyze_text
from .humanizer import humanize_content
from .models import (
    AnalyzeRequest,
    AnalyzeResponse,
    HumanizeRequest,
    HumanizeResponse,
    OverallResult,
)
from .suggestions import generate_suggestions

app = FastAPI(title="AI Content Detection API", version="1.0.0")

frontend_origins = [
    origin.strip()
    for origin in os.environ.get(
        "FRONTEND_ORIGIN",
        "http://localhost:3000,http://127.0.0.1:3000,http://localhost:3001,http://127.0.0.1:3001",
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=frontend_origins,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1):300[01]",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(payload: AnalyzeRequest):
    text = payload.content.strip()
    if not text:
        raise HTTPException(status_code=400, detail="content must not be empty")

    result = analyze_text(text)

    suggestions = []
    if payload.include_suggestions:
        try:
            suggestions = generate_suggestions(
                result["sentence_scores"], result["detected_patterns"]
            )
        except RuntimeError:
            # GROQ_API_KEY missing — degrade gracefully, scoring still works
            suggestions = []

    return AnalyzeResponse(
        overall=OverallResult(
            ai_writing_likelihood=result["overall_pct"],
            confidence=result["confidence"],
        ),
        detected_patterns=result["detected_patterns"],
        sentence_scores=result["sentence_scores"],
        highlighted_phrases=result["highlighted_phrases"],
        suggestions=suggestions,
    )


@app.post("/humanize", response_model=HumanizeResponse)
def humanize(payload: HumanizeRequest):
    try:
        humanized = humanize_content(payload.content)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Humanization failed ({type(e).__name__}). "
                   f"Try again or check GROQ_API_KEY / model name.",
        )

    if not humanized:
        raise HTTPException(status_code=502, detail="Humanization returned empty output.")

    return HumanizeResponse(humanized_content=humanized)
