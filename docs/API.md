# API Guide

Start the FastAPI backend:

```bash
uvicorn backend.app.main:app --reload
```

Open interactive API docs:

```text
http://127.0.0.1:8000/docs
```

## Health Check

```http
GET /api/health
```

Example response:

```json
{
  "status": "ok",
  "redis": "memory-fallback",
  "knowledge_base_ready": false,
  "details": {
    "ready": false,
    "total_documents": 0,
    "chroma_dir": "chroma_db",
    "last_error": null
  }
}
```
`redis` values:

- `connected`: Redis is enabled and reachable.
- `memory-fallback`: Redis is not configured or unavailable, so the backend uses local memory.
- `disabled`: Redis URL is not configured before startup.

## Chat

```http
POST /api/chat
```

Request:

```json
{
  "message": "最近睡眠浅，适合怎样调理？",
  "session_id": null,
  "user_profile": {
    "age": "25",
    "gender": "保密",
    "health": "经常熬夜"
  },
  "image_path": null
}
```

Response:

```json
{
  "session_id": "generated-session-id",
  "answer": "模型回答内容",
  "cached": false,
  "image_description": null,
  "sources": []
}
```

## Streaming Chat

```http
POST /api/chat/stream
```

The endpoint returns Server-Sent Events:

```text
event: session
data: {"session_id":"..."}

event: token
data: {"text":"..."}

event: done
data: {"session_id":"...","answer":"..."}
```

## Knowledge Base

Upload files only:

```http
POST /api/knowledge/upload
```

Upload and build the knowledge base:

```http
POST /api/knowledge/build
```

Check knowledge base status:

```http
GET /api/knowledge/status
```

Supported file types:

- `.pdf`
- `.docx`
- `.txt`
- `.md`

## Image Analysis

```http
POST /api/image/analyze
```

Request:

```json
{
  "image_path": "C:/path/to/image.jpg"
}
```

Response:

```json
{
  "description": "图片描述",
  "success": true
}
```
