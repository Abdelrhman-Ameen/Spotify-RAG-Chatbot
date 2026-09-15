from pathlib import Path

from spotify_rag.assistant import SpotifyAssistant
from spotify_rag.config import Settings


def settings() -> Settings:
    base = Path(__file__).resolve().parents[1]
    return Settings(
        base_dir=base, knowledge_path=base / "data" / "spotify_knowledge_base.json",
        chroma_path=base / "tests" / ".chroma", collection_name="test",
        embedding_model="unused", retrieval_backend="tfidf", llm_backend="extractive",
        local_llm_model="unused", ollama_model="unused",
        ollama_url="http://localhost:11434", relevance_threshold=0.05,
    )


def build_assistant() -> SpotifyAssistant:
    assistant = SpotifyAssistant(settings())
    assistant.initialize()
    return assistant


def test_offline_answer_is_grounded_and_cited():
    result = build_assistant().chat("How many songs can I download for offline listening?")
    assert result["grounded"] is True
    assert result["intent"] == "playback"
    assert "10,000" in result["response"]
    assert result["sources"][0]["url"].startswith("https://support.spotify.com/")


def test_unknown_question_refuses_to_guess():
    result = build_assistant().chat("What is the tensile strength of lunar concrete?")
    assert result["grounded"] is False
    assert "verified" in result["response"]
    assert result["sources"] == []


def test_negative_security_message_is_priority_routed():
    result = build_assistant().chat("My account was hacked and I am angry")
    assert result["intent"] == "account_security"
    assert result["sentiment"] == "negative"
    assert result["escalated"] is True


def test_small_talk_skips_retrieval():
    result = build_assistant().chat("Hello")
    assert result["intent"] == "greeting"
    assert result["sources"] == []
