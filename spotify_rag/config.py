from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    base_dir: Path
    knowledge_path: Path
    chroma_path: Path
    collection_name: str
    embedding_model: str
    retrieval_backend: str
    llm_backend: str
    local_llm_model: str
    ollama_model: str
    ollama_url: str
    sentiment_backend: str
    sentiment_model: str
    relevance_threshold: float

    @classmethod
    def from_env(cls, base_dir: Path) -> "Settings":
        try:
            from dotenv import load_dotenv

            load_dotenv(base_dir / ".env")
        except ImportError:
            pass
        return cls(
            base_dir=base_dir,
            knowledge_path=base_dir / "data" / "spotify_knowledge_base.json",
            chroma_path=base_dir / os.getenv("CHROMA_PATH", "chroma_db"),
            collection_name=os.getenv("CHROMA_COLLECTION", "spotify_support_v1"),
            embedding_model=os.getenv(
                "EMBEDDING_MODEL",
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            ),
            retrieval_backend=os.getenv("RAG_RETRIEVAL_BACKEND", "chroma").lower(),
            llm_backend=os.getenv("LLM_BACKEND", "transformers").lower(),
            local_llm_model=os.getenv("LOCAL_LLM_MODEL", "google/flan-t5-small"),
            ollama_model=os.getenv("OLLAMA_MODEL", "qwen2.5:3b"),
            ollama_url=os.getenv("OLLAMA_URL", "http://localhost:11434"),
            sentiment_backend=os.getenv("SENTIMENT_BACKEND", "transformers").lower(),
            sentiment_model=os.getenv(
                "SENTIMENT_MODEL",
                "cardiffnlp/twitter-roberta-base-sentiment-latest",
            ),
            relevance_threshold=float(os.getenv("RAG_RELEVANCE_THRESHOLD", "0.20")),
        )
