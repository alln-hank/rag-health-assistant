from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, UploadFile

from backend.app.config import settings
from backend.app.schemas import BuildKnowledgeResponse, KnowledgeStatusResponse, UploadResponse
from backend.app.services.rag_service import rag_service


router = APIRouter(prefix="/knowledge", tags=["knowledge"])


def _safe_upload_path(filename: str) -> Path:
    suffix = Path(filename).suffix.lower()
    safe_name = f"{uuid4().hex}{suffix}"
    return settings.upload_dir / safe_name


@router.post("/upload", response_model=UploadResponse)
async def upload_files(files: list[UploadFile] = File(...)):
    saved_files = []
    for file in files:
        path = _safe_upload_path(file.filename or "upload")
        path.write_bytes(await file.read())
        saved_files.append(str(path))
    return UploadResponse(files=saved_files)


@router.post("/build", response_model=BuildKnowledgeResponse)
async def build_knowledge(files: list[UploadFile] = File(...)):
    saved_files = []
    for file in files:
        path = _safe_upload_path(file.filename or "upload")
        path.write_bytes(await file.read())
        saved_files.append(str(path))

    result = rag_service.build_knowledge_base(saved_files)
    return BuildKnowledgeResponse(**result)


@router.get("/status", response_model=KnowledgeStatusResponse)
async def knowledge_status():
    return KnowledgeStatusResponse(**rag_service.get_status())
