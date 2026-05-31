# Redis Notes

Redis is optional in this project.

If `REDIS_URL` is not configured, the FastAPI backend automatically falls back to in-memory rate limiting and cache. This is enough for local development and demos.

## What Redis Is Used For

- API rate limiting.
- Answer cache for repeated text-only questions.

Redis is currently used by the FastAPI backend only. The Gradio entry `python main.py` still uses its original in-process state.

## Enable Redis

Add this to `.env`:

```env
REDIS_URL=redis://localhost:6379/0
RATE_LIMIT=10
RATE_WINDOW_SECONDS=60
CACHE_TTL_SECONDS=3600
```

Restart the backend:

```bash
uvicorn backend.app.main:app --reload
```

Then open:

```text
http://127.0.0.1:8000/api/health
```

If Redis is active, the response contains:

```json
{
  "redis": "connected"
}
```

If Redis is not active, the response contains:

```json
{
  "redis": "memory-fallback"
}
```

## Is Redis Required?

No. The project runs without Redis.

Use Redis when you want a more production-like backend. Skip Redis when you only need local development or a course demo.
