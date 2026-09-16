from __future__ import annotations

import re

from .llm import OpenSourceGenerator
from .nlp import SentimentClassifier, detect_explicit_intent, detect_intent, detect_language
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

    def chat(self, message: str, top_k: int = 4, history: list[dict] | None = None) -> dict:
        if not message:
            raise ValueError("Message cannot be empty")
        contextual_message = self._contextualize(message, history or [])
        sentiment = self.sentiment_classifier.predict(contextual_message)
        language = detect_language(message)
        explicit_intent = detect_explicit_intent(contextual_message)
        intent = explicit_intent or detect_intent(contextual_message)
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

        category_filters = {
            "account_management": "account_management",
            "account_security": "account_security",
            "login_help": "login_help",
            "billing_and_refunds": "billing",
            "refund_request": "refunds",
            "duplicate_charge": "duplicate_charge",
            "canceled_but_charged": "canceled_charge",
            "unknown_charge": "unknown_charge",
            "payment_details": "payment_details",
            "failed_payment": "failed_payment",
            "cancel_premium": "cancel_premium",
            "family_plan": "family_plan",
            "duo_plan": "duo_plan",
            "student_plan": "student_plan",
            "basic_plan": "basic_plan",
            "contact_support": "contact_support",
            "data_usage": "data_usage",
            "premium_benefits": "premium_benefits",
            "premium_pricing": "premium_pricing",
            "premium_subscription": "premium_subscription",
            "spotify_basics": "spotify_basics",
            "offline_listening": "offline_listening",
            "audio_quality": "audio_quality",
            "lossless_audio": "lossless_audio",
            "connection_issue": "connection_issue",
            "playback_issue": "playback_issue",
            "no_sound": "no_sound",
            "storage_help": "storage_help",
            "reset_password": "reset_password",
            "delete_account": "delete_account",
            "playlist_recovery": "playlist_recovery",
            "collaborative_playlist": "collaborative_playlist",
            "spotify_connect": "spotify_connect",
            "private_listening": "private_listening",
            "privacy_data": "privacy_data",
        }
        # Hard category filters are safe only for an explicit phrase match. A
        # statistical fallback prediction still searches the full knowledge base.
        category = category_filters.get(intent) if explicit_intent else None
        documents = self.retriever.search(contextual_message, top_k, category=category)
        threshold = self.settings.relevance_threshold
        if self.retriever.backend_name.startswith("tfidf"):
            threshold = min(threshold, 0.10)
        if category:
            # An explicit intent route is already a high-confidence semantic match.
            # The category itself may contain vocabulary absent from a short user query.
            threshold = 0.0
        relevant = [doc for doc in documents if doc.relevance >= threshold]
        escalated = intent in {
            "complaint", "account_security", "refund_request",
            "duplicate_charge", "canceled_but_charged",
        } or sentiment == "negative"
        if relevant:
            response = self.generator.generate(contextual_message, relevant, sentiment, language)
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

    @staticmethod
    def _contextualize(message: str, history: list[dict]) -> str:
        normalized = " ".join(message.lower().split())
        follow_up = bool(re.search(
            r"\b(that|it|this|those|them)\b|^(why|how|what about|and why|and how)\??$",
            normalized,
        ))
        if not follow_up:
            return message
        previous_user = next(
            (turn.get("content", "") for turn in reversed(history) if turn.get("role") == "user"),
            "",
        )
        if not previous_user:
            return message
        return f"Previous customer question: {previous_user}. Follow-up: {message}"
