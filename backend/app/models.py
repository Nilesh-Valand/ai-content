from typing import List, Optional, Literal
from pydantic import BaseModel, Field, field_validator


class User(BaseModel):
    id: int
    name: str
    created_at: str


class UserCreate(BaseModel):
    name: str = Field(..., max_length=100)

    @field_validator("name", mode="before")
    @classmethod
    def validate_name(cls, value):
        if value is None:
            raise ValueError("user name is required")
        if not isinstance(value, str):
            raise ValueError("user name must be a string")
        value = value.strip()
        if not value:
            raise ValueError("user name must not be empty")
        return value


class AnalyzeRequest(BaseModel):
    content: str = Field(..., max_length=20000)
    include_suggestions: bool = True
    user_id: Optional[int] = None

    @field_validator("content", mode="before")
    @classmethod
    def validate_content(cls, value):
        if value is None:
            raise ValueError("content is required")
        if not isinstance(value, str):
            raise ValueError("content must be a string")

        value = value.strip()
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
    id: int
    overall: OverallResult
    detected_patterns: List[DetectedPattern]
    sentence_scores: List[SentenceScore]
    highlighted_phrases: List[str]
    suggestions: List[Suggestion]


class Profile(BaseModel):
    id: int
    name: str
    description: str = ""
    phrase_count: int = 0
    created_at: str


class ProfileCreate(BaseModel):
    name: str = Field(..., max_length=100)
    description: Optional[str] = Field("", max_length=500)
    user_id: Optional[int] = None

    @field_validator("name", mode="before")
    @classmethod
    def validate_name(cls, value):
        if value is None:
            raise ValueError("profile name is required")
        if not isinstance(value, str):
            raise ValueError("profile name must be a string")
        value = value.strip()
        if not value:
            raise ValueError("profile name must not be empty")
        return value


class ProfileUpdate(BaseModel):
    name: str = Field(..., max_length=100)
    description: Optional[str] = Field("", max_length=500)

    @field_validator("name", mode="before")
    @classmethod
    def validate_name(cls, value):
        if value is None:
            raise ValueError("profile name is required")
        if not isinstance(value, str):
            raise ValueError("profile name must be a string")
        value = value.strip()
        if not value:
            raise ValueError("profile name must not be empty")
        return value


class HumanizeRequest(BaseModel):
    content: str = Field(..., max_length=20000)
    project_id: Optional[int] = None
    profile_id: Optional[int] = None
    phrase_ids: Optional[List[int]] = None

    @field_validator("content", mode="before")
    @classmethod
    def validate_content(cls, value):
        if value is None:
            raise ValueError("content is required")
        if not isinstance(value, str):
            raise ValueError("content must be a string")

        value = value.strip()
        if not value:
            raise ValueError("content must not be empty")

        return value


class HumanizeResponse(BaseModel):
    humanized_content: str


class TrainedPhraseCreate(BaseModel):
    ai_phrase: str = Field(..., max_length=2000)
    humanized_phrase: str = Field(..., max_length=2000)
    profile_id: Optional[int] = None

    @field_validator("ai_phrase", "humanized_phrase", mode="before")
    @classmethod
    def validate_non_empty(cls, value):
        if value is None:
            raise ValueError("this field is required")
        if not isinstance(value, str):
            raise ValueError("this field must be a string")

        value = value.strip()
        if not value:
            raise ValueError("this field must not be empty")

        return value


class TrainedPhraseUpdate(BaseModel):
    ai_phrase: str = Field(..., max_length=2000)
    humanized_phrase: str = Field(..., max_length=2000)
    profile_id: Optional[int] = None

    @field_validator("ai_phrase", "humanized_phrase", mode="before")
    @classmethod
    def validate_non_empty(cls, value):
        if value is None:
            raise ValueError("this field is required")
        if not isinstance(value, str):
            raise ValueError("this field must be a string")

        value = value.strip()
        if not value:
            raise ValueError("this field must not be empty")

        return value


class TrainedPhrase(BaseModel):
    id: int
    ai_phrase: str
    humanized_phrase: str
    profile_id: Optional[int] = None
    created_at: str


class ProjectSummary(BaseModel):
    id: int
    content: str
    ai_writing_likelihood: float
    confidence: Literal["Low", "Medium", "High"]
    has_humanized: bool
    created_at: str


class ProjectDetail(BaseModel):
    id: int
    content: str
    overall: OverallResult
    detected_patterns: List[DetectedPattern]
    sentence_scores: List[SentenceScore]
    highlighted_phrases: List[str]
    suggestions: List[Suggestion]
    humanized_content: Optional[str] = None
    created_at: str
