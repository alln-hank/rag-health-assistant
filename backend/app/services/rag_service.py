import warnings
from pathlib import Path

import jieba
from dashscope import TextReRank
from langchain_core._api.deprecation import LangChainDeprecationWarning
from langchain_core.documents import Document
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

from backend.app.config import settings


warnings.filterwarnings("ignore", message="The class `Chroma` was deprecated.*")
warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)


class RAGService:
    def __init__(self) -> None:
        self._embeddings: DashScopeEmbeddings | None = None
        self.db: Chroma | None = None
        self.bm25_index: BM25Okapi | None = None
        self.bm25_docs: list[str] = []
        self.last_error: str | None = None

    @property
    def embeddings(self) -> DashScopeEmbeddings:
        if self._embeddings is None:
            self._embeddings = DashScopeEmbeddings(model=settings.embedding_model)
        return self._embeddings

    def load_existing(self) -> None:
        if not settings.chroma_dir.exists():
            return
        if not settings.dashscope_api_key:
            self.last_error = "未设置 DASHSCOPE_API_KEY，无法加载需要 Embedding 的知识库。"
            return
        try:
            self.db = Chroma(
                persist_directory=str(settings.chroma_dir),
                embedding_function=self.embeddings,
            )
            self.rebuild_bm25()
            self.last_error = None
        except Exception as exc:
            self.db = None
            self.bm25_index = None
            self.bm25_docs = []
            self.last_error = str(exc)

    def get_status(self) -> dict:
        return {
            "ready": self.db is not None,
            "total_documents": len(self.bm25_docs),
            "chroma_dir": str(settings.chroma_dir),
            "last_error": self.last_error,
        }

    def build_knowledge_base(self, file_paths: list[str]) -> dict:
        if not settings.dashscope_api_key:
            self.last_error = "未设置 DASHSCOPE_API_KEY，无法构建知识库。"
            return {"message": self.last_error, "chunks": 0, "total_documents": len(self.bm25_docs)}
        if not file_paths:
            return {"message": "请先上传文件。", "chunks": 0, "total_documents": len(self.bm25_docs)}

        documents: list[Document] = []
        for file_path in file_paths:
            path = Path(file_path)
            if not path.exists():
                continue
            suffix = path.suffix.lower()
            if suffix == ".pdf":
                docs = PyPDFLoader(str(path)).load()
            elif suffix == ".docx":
                docs = Docx2txtLoader(str(path)).load()
            elif suffix in {".txt", ".md"}:
                docs = TextLoader(str(path), encoding="utf-8").load()
            else:
                continue
            documents.extend(docs)

        if not documents:
            return {"message": "未读取到可构建知识库的文档。", "chunks": 0, "total_documents": len(self.bm25_docs)}

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", "。", "；"],
        )
        texts = splitter.split_documents(documents)

        if self.db is not None:
            self.db.add_documents(texts)
        else:
            self.db = Chroma.from_documents(
                texts,
                self.embeddings,
                persist_directory=str(settings.chroma_dir),
            )

        self.rebuild_bm25()
        return {
            "message": "知识库更新完成。",
            "chunks": len(texts),
            "total_documents": len(self.bm25_docs),
        }

    def rebuild_bm25(self) -> None:
        if self.db is None:
            return
        all_data = self.db.get(include=["documents"])
        docs = [doc for doc in all_data.get("documents", []) if doc]
        self.bm25_docs = docs
        self.bm25_index = BM25Okapi([list(jieba.cut(doc)) for doc in docs]) if docs else None

    def retrieve_docs(self, query: str) -> list[Document]:
        if self.db is None or not query.strip():
            return []
        if self.bm25_index is None or not self.bm25_docs:
            candidates = self.db.similarity_search(query, k=settings.hybrid_top_k)
        else:
            candidates = self.hybrid_search(query, top_k=settings.hybrid_top_k)
        return self.rerank(query, candidates, top_k=settings.rerank_top_k)

    def hybrid_search(self, query: str, top_k: int) -> list[Document]:
        if self.db is None or self.bm25_index is None:
            return []

        vector_results = self.db.similarity_search_with_score(query, k=top_k)
        tokenized_query = list(jieba.cut(query))
        bm25_scores = self.bm25_index.get_scores(tokenized_query)
        if len(bm25_scores) == 0:
            return [doc for doc, _ in vector_results[:top_k]]

        top_bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_k]
        doc_map = {}
        for doc, score in vector_results:
            content = doc.page_content
            vec_sim = 1.0 / (1.0 + score)
            doc_map[content] = {"doc": doc, "vector_score": vec_sim, "bm25_score": 0.0}

        max_bm25 = bm25_scores.max() if bm25_scores.max() > 0 else 1.0
        for idx in top_bm25_indices:
            content = self.bm25_docs[idx]
            bm25_norm = bm25_scores[idx] / max_bm25
            if content in doc_map:
                doc_map[content]["bm25_score"] = bm25_norm
            else:
                doc_map[content] = {
                    "doc": Document(page_content=content),
                    "vector_score": 0.0,
                    "bm25_score": bm25_norm,
                }

        results = []
        for scores in doc_map.values():
            final_score = (1 - settings.bm25_weight) * scores["vector_score"] + settings.bm25_weight * scores["bm25_score"]
            results.append((scores["doc"], final_score))
        results.sort(key=lambda item: item[1], reverse=True)
        return [doc for doc, _ in results[:top_k]]

    def rerank(self, query: str, docs: list[Document], top_k: int) -> list[Document]:
        if not docs:
            return []
        documents = [doc.page_content for doc in docs]
        try:
            resp = TextReRank.call(
                model=settings.rerank_model,
                query=query,
                documents=documents,
                top_n=top_k,
            )
            if resp.status_code == 200:
                indices = [item["index"] for item in resp.output["results"]]
                return [docs[i] for i in indices]
        except Exception:
            return docs[:top_k]
        return docs[:top_k]


rag_service = RAGService()
