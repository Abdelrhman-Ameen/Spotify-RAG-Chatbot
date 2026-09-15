from __future__ import annotations

from .llm import OpenSourceGenerator
from .nlp import SentimentClassifier, detect_intent, detect_language
from .retrieval import SpotifyRetriever

SMALL_TALK = {
    "greeting": "Hi! I’m the Spotify Support assistant. Ask me about accounts, plans, playback, downloads, playlists, devices, privacy, podcasts, or audiobooks.",
    "gratitude": "You’re welcome! Is there anything else you’d like to know about Spotify?",
    "goodbye": "Goodbye, and happy listening!",
}

OPINION_RESPONSES = {
    "positive": "I’m glad you’re enjoying Spotify! If you want, I can also help with Premium, playlists, downloads, audio quality, or account settings.",
    "negative": "I’m sorry Spotify hasn’t met your expectations. Tell me what went wrong—such as playback, billing, your account, or a missing feature—and I’ll help with the right steps.",
    "neutral": "Thanks for sharing your thoughts about Spotify. Tell me more, or ask me about an account, plan, playback, playlist, or device issue.",
}


class SpotifyAssistant:
    def __init__(self, settings):
        self.settings = settings
        self.retriever = SpotifyRetriever(settings)
        self.generator = OpenSourceGenerator(settings)
        self.sentiment_classifier = SentimentClassifier(
            settings.sentiment_backend, settings.sentiment_model
        )

    def initialize(self) -> None:
        self.retriever.initialize()
        self.sentiment_classifier.initialize()
        self.generator.initialize()

    def chat(self, message: str, top_k: int = 4) -> dict:
        if not message:
            raise ValueError("Message cannot be empty")
        sentiment = self.sentiment_classifier.predict(message)
        language = detect_language(message)
        intent = detect_intent(message)
        if intent in SMALL_TALK:
            return {
                "response": SMALL_TALK[intent], "language": language,
                "sentiment": sentiment, "intent": intent, "escalated": False,
                "grounded": True, "sources": [],
            }
        if intent == "opinion":
            return {
                "response": OPINION_RESPONSES[sentiment], "language": language,
                "sentiment": sentiment, "intent": intent, "escalated": False,
                "grounded": True, "sources": [],
            }

        documents = self.retriever.search(message, top_k)
        threshold = self.settings.relevance_threshold
        if self.retriever.backend_name.startswith("tfidf"):
            threshold = min(threshold, 0.10)
        relevant = [doc for doc in documents if doc.relevance >= threshold]
        escalated = intent in {"complaint", "account_security"} or sentiment == "negative"
        if relevant:
            response = self.generator.generate(message, relevant, sentiment, language)
            grounded = True
        else:
            response = (
                "I couldn’t match that to a verified Spotify help topic yet. "
                "Try rephrasing it with the feature or problem involved, or use the official "
                "Spotify Support contact page for account-specific help."
            )
            grounded = False
        return {
            "response": response, "language": language, "sentiment": sentiment,
            "intent": intent, "escalated": escalated, "grounded": grounded,
            "sources": [{
                "title": doc.title, "url": doc.url, "relevance": round(doc.relevance, 3)
            } for doc in relevant[:3]],
        }
