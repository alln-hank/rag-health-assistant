from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from backend.app.schemas import ChatRequest, ChatResponse, SessionSummary
from backend.app.services.chat_service import chat_service


router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest):
    return await chat_service.answer(request)


@router.post("/stream")
async def chat_stream(request: ChatRequest):
    return StreamingResponse(chat_service.stream_answer(request), media_type="text/event-stream")


@router.get("/sessions", response_model=list[SessionSummary])
async def list_sessions():
    return chat_service.list_sessions()


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    deleted = chat_service.delete_session(session_id)
    return {"deleted": deleted}
