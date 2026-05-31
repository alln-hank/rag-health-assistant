from fastapi import APIRouter

from backend.app.schemas import ToolRunRequest, ToolRunResponse
from backend.app.services.health_tool_service import health_tool_service


router = APIRouter(prefix="/tools", tags=["tools"])


@router.get("")
async def list_tools():
    return {"tools": health_tool_service.list_tools()}


@router.post("/run", response_model=ToolRunResponse)
async def run_tools(request: ToolRunRequest):
    results = health_tool_service.run_tools(request.text, request.user_profile)
    return ToolRunResponse(results=results)
