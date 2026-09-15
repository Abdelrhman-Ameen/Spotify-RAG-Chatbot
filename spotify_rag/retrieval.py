from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RetrievedDocument:
    id: str
    title: str
    answer: str
    url: str
    category: str
    relevance: float


def load_entries(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        entries = json.load(stream).get("entries", [])
    if not entries:
        raise ValueError(f"Knowledge base has no entries: {path}")
    return entries


def entry_document(entry: dict[str, Any]) -> str:
    questions = " ".join(entry.get("questions", []))
    return f"{entry['title']}\nCommon questions: {questions}\nOfficial answer: {entry['answer']}"


class SpotifyRetriever:
    def __init__(self, settings):
        self.settings = settings
        self.entries = load_entries(settings.knowledge_path)
        self.backend_name = settings.retrieval_backend
        self._collection = None
        self._tfidf = None
        self._matrix = None

    @property
    def count(self) -> int:
        return len(self.entries)

    def initialize(self) -> None:
        if self.backend_name == "tfidf":
            self._initialize_tfidf()
            return
        if self.backend_name != "chroma":
            raise ValueError("RAG_RETRIEVAL_BACKEND must be 'chroma' or 'tfidf'")
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
            from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

            embedding_function = SentenceTransformerEmbeddingFunction(model_name=self.settings.embedding_model)
            client = chromadb.PersistentClient(
                path=str(self.settings.chroma_path),
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            self._collection = client.get_or_create_collection(
                name=self.settings.collection_name,
                embedding_function=embedding_function,
                metadata={"hnsw:space": "cosine"},
            )
            existing = set(self._collection.get(include=[])["ids"])
            current = {entry["id"] for entry in self.entries}
            if existing - current:
                self._collection.delete(ids=list(existing - current))
            self._upsert(self.entries)
        except Exception as exc:
            print(f"Chroma initialization failed ({exc}); using local TF-IDF retrieval.")
            self.backend_name = "tfidf-fallback"
            self._collection = None
            self._initialize_tfidf()

    def _upsert(self, entries: list[dict[str, Any]]) -> None:
        self._collection.upsert(
            ids=[entry["id"] for entry in entries],
            documents=[entry_document(entry) for entry in entries],
            metadatas=[{
                "title": entry["title"], "answer": entry["answer"],
                "url": entry["source_url"], "category": entry["category"],
            } for entry in entries],
        )

    def _initialize_tfidf(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._tfidf = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        self._matrix = self._tfidf.fit_transform(entry_document(entry) for entry in self.entries)

    def search(self, query: str, top_k: int = 4, category: str | None = None) -> list[RetrievedDocument]:
        if self._collection is not None:
            query_args = {
                "query_texts": [query], "n_results": min(top_k, self.count),
                "include": ["metadatas", "distances"],
            }
            if category:
                query_args["where"] = {"category": category}
            result = self._collection.query(**query_args)
            return [RetrievedDocument(
                id=doc_id, title=metadata["title"], answer=metadata["answer"],
                url=metadata["url"], category=metadata["category"],
                relevance=max(0.0, min(1.0, 1.0 - float(distance))),
            ) for doc_id, metadata, distance in zip(
                result["ids"][0], result["metadatas"][0], result["distances"][0]
            )]
        if self._tfidf is None:
            raise RuntimeError("Retriever has not been initialized")
        from sklearn.metrics.pairwise import cosine_similarity

        scores = cosine_similarity(self._tfidf.transform([query]), self._matrix)[0]
        ranked = scores.argsort()[::-1]
        if category:
            ranked = [index for index in ranked if self.entries[index]["category"] == category]
        indices = ranked[:top_k]
        return [RetrievedDocument(
            id=self.entries[index]["id"], title=self.entries[index]["title"],
            answer=self.entries[index]["answer"], url=self.entries[index]["source_url"],
            category=self.entries[index]["category"], relevance=float(scores[index]),
        ) for index in indices]
