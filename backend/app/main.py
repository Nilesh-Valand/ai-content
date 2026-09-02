import os
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

from .analyzer import analyze_text
from . import db
from .humanizer import humanize_content
from .models import (
    AnalyzeRequest,
    AnalyzeResponse,
    HumanizeRequest,
    HumanizeResponse,
    OverallResult,
    ProjectDetail,
    ProjectSummary,
    TrainedPhrase,
    TrainedPhraseCreate,
    TrainedPhraseUpdate,
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

    project = db.create_project(
        content=text,
        overall_pct=result["overall_pct"],
        confidence=result["confidence"],
        detected_patterns=[p.model_dump() for p in result["detected_patterns"]],
        sentence_scores=[s.model_dump() for s in result["sentence_scores"]],
        highlighted_phrases=result["highlighted_phrases"],
        suggestions=[s.model_dump() for s in suggestions],
    )

    return AnalyzeResponse(
        id=project["id"],
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

    if payload.project_id is not None:
        db.update_project_humanized(payload.project_id, humanized)

    return HumanizeResponse(humanized_content=humanized)


@app.post("/train/phrases", response_model=TrainedPhrase)
def create_phrase(payload: TrainedPhraseCreate):
    row = db.create_trained_phrase(payload.ai_phrase, payload.humanized_phrase)
    return TrainedPhrase(**row)


@app.get("/train/phrases", response_model=List[TrainedPhrase])
def get_phrases():
    return [TrainedPhrase(**row) for row in db.list_trained_phrases()]


@app.put("/train/phrases/{phrase_id}", response_model=TrainedPhrase)
def update_phrase(phrase_id: int, payload: TrainedPhraseUpdate):
    row = db.update_trained_phrase(phrase_id, payload.ai_phrase, payload.humanized_phrase)
    if row is None:
        raise HTTPException(status_code=404, detail="Phrase not found")
    return TrainedPhrase(**row)


@app.delete("/train/phrases/{phrase_id}")
def remove_phrase(phrase_id: int):
    deleted = db.delete_trained_phrase(phrase_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Phrase not found")
    return {"status": "deleted", "id": phrase_id}


@app.get("/projects", response_model=List[ProjectSummary])
def get_projects():
    return [
        ProjectSummary(
            id=row["id"],
            content=row["content"],
            ai_writing_likelihood=row["overall_pct"],
            confidence=row["confidence"],
            has_humanized=row["humanized_content"] is not None,
            created_at=row["created_at"],
        )
        for row in db.list_projects()
    ]


@app.get("/projects/{project_id}", response_model=ProjectDetail)
def get_project(project_id: int):
    row = db.get_project(project_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Project not found")
    return ProjectDetail(
        id=row["id"],
        content=row["content"],
        overall=OverallResult(
            ai_writing_likelihood=row["overall_pct"], confidence=row["confidence"]
        ),
        detected_patterns=row["detected_patterns"],
        sentence_scores=row["sentence_scores"],
        highlighted_phrases=row["highlighted_phrases"],
        suggestions=row["suggestions"],
        humanized_content=row["humanized_content"],
        created_at=row["created_at"],
    )


@app.delete("/projects/{project_id}")
def remove_project(project_id: int):
    deleted = db.delete_project(project_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Project not found")
    return {"status": "deleted", "id": project_id}
