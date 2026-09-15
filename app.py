"""FastAPI entry point for the Spotify Support RAG assistant."""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from spotify_rag.assistant import SpotifyAssistant
from spotify_rag.config import Settings

BASE_DIR = Path(__file__).resolve().parent


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=4, ge=1, le=8)
    history: list[ChatTurn] = Field(default_factory=list, max_length=10)


class Source(BaseModel):
    title: str
    url: str
    relevance: float


class ChatResponse(BaseModel):
    response: str
    language: str
    sentiment: str
    intent: str
    escalated: bool = False
    grounded: bool = True
    sources: list[Source] = Field(default_factory=list)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.assistant = SpotifyAssistant(Settings.from_env(BASE_DIR))
    app.state.assistant.initialize()
    yield


app = FastAPI(
    title="Spotify Support RAG Assistant",
    description="A source-grounded Spotify help chatbot powered by a local open-source LLM.",
    version="2.0.0",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


@app.get("/", include_in_schema=False)
async def root() -> FileResponse:
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.get("/health")
async def health() -> dict:
    assistant = app.state.assistant
    return {
        "status": "ok",
        "knowledge_entries": assistant.retriever.count,
        "retrieval_backend": assistant.retriever.backend_name,
        "llm_backend": assistant.generator.backend_name,
        "llm_ready": assistant.generator.ready,
        "sentiment_backend": assistant.sentiment_classifier.backend_name,
        "sentiment_ready": assistant.sentiment_classifier.ready,
    }


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest) -> ChatResponse:
    try:
        history = [turn.model_dump() for turn in request.history]
        return ChatResponse(**app.state.assistant.chat(request.message.strip(), request.top_k, history))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"The assistant is temporarily unavailable: {exc}") from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
