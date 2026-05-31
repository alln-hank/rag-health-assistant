from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.config import settings
from backend.app.routers import chat, image, knowledge
from backend.app.schemas import HealthResponse
from backend.app.services.cache_service import cache_service
from backend.app.services.rag_service import rag_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    await cache_service.connect()
    rag_service.load_existing()
    yield
    await cache_service.close()


app = FastAPI(
    title=settings.app_name,
    description="健康养生 RAG 智能问答助手后端 API",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    public_paths = {"/", "/api/health", "/docs", "/redoc", "/openapi.json"}
    if request.url.path in public_paths:
        return await call_next(request)

    client_id = request.client.host if request.client else "unknown"
    allowed = await cache_service.allow_request(client_id, scope=request.url.path)
    if not allowed:
        return JSONResponse(
            status_code=429,
            content={"detail": "请求过于频繁，请稍后再试。"},
        )
    return await call_next(request)


@app.get("/")
async def root():
    return {
        "name": settings.app_name,
        "docs": "/docs",
        "health": f"{settings.api_prefix}/health",
    }


@app.get(f"{settings.api_prefix}/health", response_model=HealthResponse)
async def health():
    status = rag_service.get_status()
    return HealthResponse(
        status="ok",
        redis=cache_service.redis_status,
        knowledge_base_ready=status["ready"],
        details=status,
    )


app.include_router(chat.router, prefix=settings.api_prefix)
app.include_router(knowledge.router, prefix=settings.api_prefix)
app.include_router(image.router, prefix=settings.api_prefix)
