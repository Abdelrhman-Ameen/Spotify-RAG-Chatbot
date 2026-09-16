from dataclasses import replace
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
        ollama_url="http://localhost:11434", sentiment_backend="heuristic",
        sentiment_model="unused", relevance_threshold=0.05,
    )


def build_assistant() -> SpotifyAssistant:
    assistant = SpotifyAssistant(settings())
    assistant.initialize()
    return assistant


def test_offline_answer_is_grounded_and_cited():
    result = build_assistant().chat("How many songs can I download for offline listening?")
    assert result["grounded"] is True
    assert result["intent"] == "offline_listening"
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


def test_positive_opinion_uses_conversation_route():
    result = build_assistant().chat("I love Spotify")
    assert result["intent"] == "opinion"
    assert result["sentiment"] == "positive"
    assert "glad" in result["response"].lower()
    assert result["sources"] == []


def test_subscribe_question_returns_basic_support_answer():
    assistant = SpotifyAssistant(replace(settings(), relevance_threshold=0.20))
    assistant.initialize()
    result = assistant.chat("Can you tell me how to subscribe?")
    assert result["grounded"] is True
    assert "Premium" in result["response"]
    assert result["sources"]


def test_why_buy_spotify_returns_benefits_not_checkout_steps():
    result = build_assistant().chat("Why do I buy Spotify?")
    assert result["intent"] == "premium_benefits"
    assert "ad-free" in result["response"]
    assert result["sources"][0]["title"] == "Spotify Premium benefits"


def test_ambiguous_follow_up_uses_previous_user_question():
    history = [
        {"role": "user", "content": "Why do I buy Spotify?"},
        {"role": "assistant", "content": "Premium includes ad-free listening."},
    ]
    result = build_assistant().chat("Why do I do that?", history=history)
    assert result["intent"] == "premium_benefits"
    assert "ad-free" in result["response"]
    assert result["grounded"] is True


def test_price_question_uses_regional_pricing_topic():
    result = build_assistant().chat("What is the price of Spotify price?")
    assert result["intent"] == "premium_pricing"
    assert "country or region" in result["response"]
    assert result["sources"][0]["title"] == "Spotify Premium prices"
