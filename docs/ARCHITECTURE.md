# Architecture

This project currently supports two runtime modes:

1. Gradio single-app mode for local visual demos.
2. FastAPI backend mode for API-based integration and future Vue/React frontends.

## Current Architecture

```text
User Browser
  |
  | Gradio UI
  v
main.py
  |
  | RAG retrieval / image analysis / chat generation
  v
DashScope + ChromaDB + BM25
```

## FastAPI Architecture

```text
Frontend or API Client
  |
  | REST / SSE
  v
FastAPI backend
  |
  +-- routers/
  |   +-- chat.py
  |   +-- knowledge.py
  |   +-- image.py
  |
  +-- services/
      +-- chat_service.py
      +-- rag_service.py
      +-- image_service.py
      +-- cache_service.py
      +-- health_tool_service.py
```

## Service Responsibilities

- `chat_service.py`: prompt construction, conversation memory, answer generation, SSE streaming.
- `rag_service.py`: document loading, text splitting, vector storage, BM25 retrieval, reranking.
- `image_service.py`: health-related image description through DashScope multimodal model.
- `health_tool_service.py`: deterministic health tools such as BMI calculation, water intake estimation, sleep planning, and exercise heart-rate estimation.
- `cache_service.py`: optional Redis-backed rate limiting and answer cache, with in-memory fallback.

## Why Keep Gradio

Gradio remains useful for quick demos and local experiments. The FastAPI backend is added as an enterprise-oriented path without breaking the original workflow.

## Next Architecture Steps

- Add a Vue3 frontend that calls the FastAPI APIs.
- Add MySQL or SQLite persistence for long-term sessions.
- Move shared RAG logic out of `main.py` so Gradio and FastAPI can reuse the same service layer.
- Expand tool calling with weather, food nutrition, and reminder tools when external APIs are available.
- Add Docker Compose for API, Redis, and optional database services.
