from typing import Any

from pydantic import BaseModel, Field


class UserProfile(BaseModel):
    age: str | None = None
    gender: str | None = "保密"
    health: str | None = None


class SourceSnippet(BaseModel):
    index: int
    content: str


class ChatRequest(BaseModel):
    message: str = Field(default="", max_length=1000)
    session_id: str | None = None
    user_profile: UserProfile | None = None
    image_path: str | None = None


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    cached: bool = False
    image_description: str | None = None
    sources: list[SourceSnippet] = Field(default_factory=list)


class UploadResponse(BaseModel):
    files: list[str]


class BuildKnowledgeRequest(BaseModel):
    file_paths: list[str]


class BuildKnowledgeResponse(BaseModel):
    message: str
    chunks: int = 0
    total_documents: int = 0


class KnowledgeStatusResponse(BaseModel):
    ready: bool
    total_documents: int
    chroma_dir: str
    last_error: str | None = None


class ImageAnalyzeRequest(BaseModel):
    image_path: str


class ImageAnalyzeResponse(BaseModel):
    description: str
    success: bool


class SessionSummary(BaseModel):
    session_id: str
    turns: int
    last_message: str | None = None


class HealthResponse(BaseModel):
    status: str
    redis: str
    knowledge_base_ready: bool
    details: dict[str, Any] = Field(default_factory=dict)
