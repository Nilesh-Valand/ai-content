from typing import List, Optional, Literal
from pydantic import BaseModel, Field, field_validator


class AnalyzeRequest(BaseModel):
    content: str = Field(..., max_length=20000)
    include_suggestions: bool = True

    @field_validator("content", mode="before")
    @classmethod
    def validate_content(cls, value):
        if value is None:
            raise ValueError("content is required")

        value = str(value).strip()
        if not value:
            raise ValueError("content must not be empty")

        return value


class DetectedPattern(BaseModel):
    pattern: str
    severity: Literal["Low", "Medium", "High"]
    score: float  # 0-1 raw signal strength
    examples: List[str] = []


class SentenceScore(BaseModel):
    index: int
    text: str
    ai_likelihood: float  # 0-100


class Suggestion(BaseModel):
    detected_sentence: Optional[str] = None
    issue: str
    suggestion: str
    improved_direction: Optional[str] = None


class OverallResult(BaseModel):
    ai_writing_likelihood: float  # 0-100
    confidence: Literal["Low", "Medium", "High"]


class AnalyzeResponse(BaseModel):
    overall: OverallResult
    detected_patterns: List[DetectedPattern]
    sentence_scores: List[SentenceScore]
    highlighted_phrases: List[str]
    suggestions: List[Suggestion]


class HumanizeRequest(BaseModel):
    content: str = Field(..., max_length=20000)

    @field_validator("content", mode="before")
    @classmethod
    def validate_content(cls, value):
        if value is None:
            raise ValueError("content is required")

        value = str(value).strip()
        if not value:
            raise ValueError("content must not be empty")

        return value


class HumanizeResponse(BaseModel):
    humanized_content: str


class TrainedPhraseCreate(BaseModel):
    ai_phrase: str = Field(..., max_length=2000)
    humanized_phrase: str = Field(..., max_length=2000)

    @field_validator("ai_phrase", "humanized_phrase", mode="before")
    @classmethod
    def validate_non_empty(cls, value):
        if value is None:
            raise ValueError("this field is required")

        value = str(value).strip()
        if not value:
            raise ValueError("this field must not be empty")

        return value


class TrainedPhrase(BaseModel):
    id: int
    ai_phrase: str
    humanized_phrase: str
    created_at: str
