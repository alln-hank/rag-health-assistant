import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


BASE_DIR = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    app_name: str = "RAG Health Assistant API"
    api_prefix: str = "/api"
    dashscope_api_key: str = os.getenv("DASHSCOPE_API_KEY", "").strip()
    chroma_dir: Path = BASE_DIR / "chroma_db"
    upload_dir: Path = BASE_DIR / "uploads"
    log_file: Path = BASE_DIR / "app.log"
    redis_url: str = os.getenv("REDIS_URL", "").strip()
    rate_limit: int = int(os.getenv("RATE_LIMIT", "10"))
    rate_window_seconds: int = int(os.getenv("RATE_WINDOW_SECONDS", "60"))
    cache_ttl_seconds: int = int(os.getenv("CACHE_TTL_SECONDS", "3600"))
    llm_model: str = os.getenv("LLM_MODEL", "qwen-max")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-v1")
    rerank_model: str = os.getenv("RERANK_MODEL", "gte-rerank")
    vision_model: str = os.getenv("VISION_MODEL", "qwen-vl-plus")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "400"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "80"))
    hybrid_top_k: int = int(os.getenv("HYBRID_TOP_K", "10"))
    rerank_top_k: int = int(os.getenv("RERANK_TOP_K", "5"))
    bm25_weight: float = float(os.getenv("BM25_WEIGHT", "0.4"))
    max_input_length: int = int(os.getenv("MAX_INPUT_LENGTH", "500"))


settings = Settings()

if settings.dashscope_api_key:
    os.environ["DASHSCOPE_API_KEY"] = settings.dashscope_api_key

settings.upload_dir.mkdir(parents=True, exist_ok=True)
