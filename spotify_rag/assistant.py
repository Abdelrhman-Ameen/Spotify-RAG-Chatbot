from __future__ import annotations

from .llm import OpenSourceGenerator
from .nlp import detect_intent, detect_language, detect_sentiment
from .retrieval import SpotifyRetriever

SMALL_TALK = {
    "greeting": "Hi! I’m the Spotify Support assistant. Ask me about accounts, plans, playback, downloads, playlists, devices, privacy, podcasts, or audiobooks.",
    "gratitude": "You’re welcome! Is there anything else you’d like to know about Spotify?",
    "goodbye": "Goodbye, and happy listening!",
}


class SpotifyAssistant:
    def __init__(self, settings):
        self.settings = settings
        self.retriever = SpotifyRetriever(settings)
        self.generator = OpenSourceGenerator(settings)

    def initialize(self) -> None:
        self.retriever.initialize()
        self.generator.initialize()

    def chat(self, message: str, top_k: int = 4) -> dict:
        if not message:
            raise ValueError("Message cannot be empty")
        language = detect_language(message)
        sentiment = detect_sentiment(message)
        intent = detect_intent(message)
        if intent in SMALL_TALK:
            return {
                "response": SMALL_TALK[intent], "language": language,
                "sentiment": sentiment, "intent": intent, "escalated": False,
                "grounded": True, "sources": [],
            }

        documents = self.retriever.search(message, top_k)
        relevant = [doc for doc in documents if doc.relevance >= self.settings.relevance_threshold]
        escalated = intent in {"complaint", "account_security"} or sentiment == "negative"
        if relevant:
            response = self.generator.generate(message, relevant, sentiment, language)
            grounded = True
        else:
            response = (
                "I don’t have enough verified Spotify information to answer that confidently. "
                "Please use the official Spotify Support contact page so an expert can help."
            )
            grounded = False
        return {
            "response": response, "language": language, "sentiment": sentiment,
            "intent": intent, "escalated": escalated, "grounded": grounded,
            "sources": [{
                "title": doc.title, "url": doc.url, "relevance": round(doc.relevance, 3)
            } for doc in relevant[:3]],
        }
