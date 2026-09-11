import os
from typing import List, Optional

import groq
import openai
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

from .analyzer import analyze_text
from . import db
from .humanizer import humanize_with_score
from .models import (
    AnalyzeRequest,
    AnalyzeResponse,
    HumanizeRequest,
    HumanizeResponse,
    OverallResult,
    Profile,
    ProfileCreate,
    ProfileUpdate,
    ProjectDetail,
    ProjectSummary,
    TrainedPhrase,
    TrainedPhraseCreate,
    TrainedPhraseUpdate,
    User,
    UserCreate,
)
from .suggestions import generate_suggestions

app = FastAPI(title="AI Content Detection API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


# --- User Endpoints ---
# No login/auth — a "user" is just a named bucket the frontend switches
# between (picked from a dropdown, remembered in the browser) to keep each
# person's profiles/phrases/history separate on a shared install.

@app.get("/users", response_model=List[User])
def get_users():
    return [User(**row) for row in db.list_users()]


@app.post("/users", response_model=User)
def create_user(payload: UserCreate):
    try:
        row = db.create_user(payload.name)
        return User(**row)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not create user: {str(e)}")


@app.delete("/users/{user_id}")
def delete_user(user_id: int):
    try:
        deleted = db.delete_user(user_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="User not found")
        return {"status": "deleted", "id": user_id}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# --- Profile Endpoints ---

@app.get("/profiles", response_model=List[Profile])
def get_profiles(user_id: Optional[int] = None):
    return [Profile(**row) for row in db.list_profiles(user_id=user_id)]


@app.post("/profiles", response_model=Profile)
def create_profile(payload: ProfileCreate):
    try:
        row = db.create_profile(payload.name, payload.description or "", payload.user_id)
        return Profile(**row)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not create profile: {str(e)}")


@app.put("/profiles/{profile_id}", response_model=Profile)
def update_profile(profile_id: int, payload: ProfileUpdate):
    row = db.update_profile(profile_id, payload.name, payload.description or "")
    if row is None:
        raise HTTPException(status_code=404, detail="Profile not found")
    return Profile(**row)


@app.delete("/profiles/{profile_id}")
def delete_profile(profile_id: int):
    deleted = db.delete_profile(profile_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Profile not found")
    return {"status": "deleted", "id": profile_id}


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
        user_id=payload.user_id,
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
        humanized, ai_score_after = humanize_with_score(
            payload.content,
            profile_id=payload.profile_id,
            phrase_ids=payload.phrase_ids,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except (groq.RateLimitError, openai.RateLimitError) as e:
        # _call_groq (humanizer.py) already retries a 429 several times
        # with backoff (and tries a fallback model, if one is configured)
        # before ever raising it this far — reaching here means the
        # provider's rate limit is still exhausted after all of that, so
        # this is a real "wait longer" case, not a bug. Surface the
        # provider's own message (which typically names the exact limit
        # type — e.g. requests-per-minute vs. tokens-per-day — and how
        # long until it resets) instead of a generic one, so this is
        # self-diagnosing without needing to dig through server logs.
        try:
            provider_detail = e.body.get("error", {}).get("message") if isinstance(e.body, dict) else None
        except Exception:
            provider_detail = None
        raise HTTPException(
            status_code=429,
            detail="LLM provider rate limit reached — the humanizer makes several AI calls "
                   "per request, so this can happen after repeated use in a short window. "
                   + (provider_detail or "Wait a bit and try again."),
        )
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Humanization failed ({type(e).__name__}). "
                   f"Try again or check GROQ_API_KEY / model name.",
        )

    if not humanized:
        raise HTTPException(status_code=502, detail="Humanization returned empty output.")

    project_id = payload.project_id
    if project_id is not None:
        db.update_project_humanized(project_id, humanized)
    else:
        # No project to attach this to (the frontend no longer requires a
        # separate /analyze step before humanizing) — create one now so it
        # still shows up in History, scored the same way /analyze scores
        # anything else. This is a local, non-LLM call (analyze_text does
        # no API calls), so it doesn't add to the humanize request's own
        # rate-limit/latency footprint.
        try:
            analysis = analyze_text(payload.content)
            project = db.create_project(
                content=payload.content,
                overall_pct=analysis["overall_pct"],
                confidence=analysis["confidence"],
                detected_patterns=[p.model_dump() for p in analysis["detected_patterns"]],
                sentence_scores=[s.model_dump() for s in analysis["sentence_scores"]],
                highlighted_phrases=analysis["highlighted_phrases"],
                suggestions=[],
                user_id=payload.user_id,
            )
            db.update_project_humanized(project["id"], humanized)
            project_id = project["id"]
        except Exception:
            # History is a convenience, not the point of this endpoint —
            # never fail a successful humanize just because saving it
            # afterward didn't work.
            project_id = None

    return HumanizeResponse(humanized_content=humanized, ai_score_after=ai_score_after, project_id=project_id)


@app.post("/train/phrases", response_model=TrainedPhrase)
def create_phrase(payload: TrainedPhraseCreate):
    row = db.create_trained_phrase(payload.ai_phrase, payload.humanized_phrase, payload.profile_id)
    return TrainedPhrase(**row)


@app.get("/train/phrases", response_model=List[TrainedPhrase])
def get_phrases(profile_id: Optional[int] = None):
    return [TrainedPhrase(**row) for row in db.list_trained_phrases(profile_id=profile_id)]


@app.put("/train/phrases/{phrase_id}", response_model=TrainedPhrase)
def update_phrase(phrase_id: int, payload: TrainedPhraseUpdate):
    row = db.update_trained_phrase(phrase_id, payload.ai_phrase, payload.humanized_phrase, payload.profile_id)
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
def get_projects(user_id: Optional[int] = None):
    return [
        ProjectSummary(
            id=row["id"],
            content=row["content"],
            ai_writing_likelihood=row["overall_pct"],
            confidence=row["confidence"],
            has_humanized=row["humanized_content"] is not None,
            created_at=row["created_at"],
        )
        for row in db.list_projects(user_id=user_id)
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
