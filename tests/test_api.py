import os

os.environ["RAG_RETRIEVAL_BACKEND"] = "tfidf"
os.environ["LLM_BACKEND"] = "extractive"
os.environ["SENTIMENT_BACKEND"] = "heuristic"
os.environ["RAG_RELEVANCE_THRESHOLD"] = "0.05"

from fastapi.testclient import TestClient
from app import app


def test_health_and_chat_contract():
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["knowledge_entries"] >= 25
        response = client.post("/chat", json={"message": "How do I cancel Premium?"})
        assert response.status_code == 200
        body = response.json()
        assert body["intent"] == "premium_plans"
        assert body["grounded"] is True
        assert body["sources"]


def test_chat_validation():
    with TestClient(app) as client:
        assert client.post("/chat", json={"message": ""}).status_code == 422


def test_chat_accepts_history_for_follow_up_questions():
    with TestClient(app) as client:
        response = client.post("/chat", json={
            "message": "Why do I do that?",
            "history": [
                {"role": "user", "content": "Why should I buy Spotify Premium?"},
                {"role": "assistant", "content": "Premium offers additional listening benefits."},
            ],
        })
        assert response.status_code == 200
        assert response.json()["intent"] == "premium_benefits"
