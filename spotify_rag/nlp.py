from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

NEGATIVE_WORDS = {
    "angry", "annoyed", "awful", "bad", "broken", "charged", "complaint",
    "disappointed", "frustrated", "hate", "horrible", "refund", "scam",
    "stolen", "terrible", "unacceptable", "upset", "worst", "wrong",
}
POSITIVE_WORDS = {
    "amazing", "awesome", "excellent", "good", "great", "happy", "love",
    "perfect", "thanks", "thank", "wonderful",
}
OPINION_PHRASES = (
    "i love spotify", "i like spotify", "spotify is great", "spotify is amazing",
    "spotify is good", "i hate spotify", "spotify is bad", "spotify is terrible",
    "my opinion", "i think spotify", "great app", "love this app", "hate this app",
)
INTENT_PATTERNS = [
    ("greeting", ("hello", "hi", "hey", "good morning", "good evening")),
    ("gratitude", ("thank you", "thanks", "appreciate")),
    ("goodbye", ("goodbye", "bye", "see you")),
    ("opinion", OPINION_PHRASES),
    ("account_security", ("hacked", "someone accessed", "stolen account", "sign out everywhere")),
    ("account_management", ("password", "log in", "login", "email address", "delete account", "username")),
    ("billing_and_refunds", ("charged", "payment", "billing", "invoice", "receipt", "refund", "gift card")),
    ("premium_pricing", ("price of spotify", "spotify price", "how much is spotify", "how much does spotify", "premium cost", "cost of spotify", "price of premium", "premium price")),
    ("premium_benefits", ("why buy spotify", "why do i buy spotify", "why would i buy spotify", "why should i buy spotify", "why get premium", "why should i get premium", "is premium worth", "premium benefits", "benefits of premium", "what do i get with premium")),
    ("premium_plans", ("premium", "family plan", "duo", "student plan", "subscription", "cancel plan")),
    ("playback", ("not playing", "no sound", "buffer", "offline", "download", "audio quality", "lossless")),
    ("playlists_and_library", ("playlist", "library", "liked songs", "recover playlist", "collaborative")),
    ("devices", ("connect", "speaker", "tv", "watch", "car", "bluetooth", "device")),
    ("privacy", ("privacy", "private session", "data", "listening activity", "public playlist")),
    ("artists_and_creators", ("artist", "podcast", "creator", "upload music", "show")),
    ("audiobooks", ("audiobook", "listening hours", "top-up")),
    ("complaint", ("complaint", "terrible", "worst", "unacceptable", "report a problem")),
]


class SentimentClassifier:
    """Three-class transformer sentiment with a deterministic offline fallback."""

    def __init__(self, backend: str, model_name: str):
        self.backend_name = backend
        self.model_name = model_name
        self.ready = backend == "heuristic"
        self._classifier = None

    def initialize(self) -> None:
        if self.backend_name == "heuristic":
            return
        if self.backend_name != "transformers":
            raise ValueError("SENTIMENT_BACKEND must be 'transformers' or 'heuristic'")
        try:
            from transformers import pipeline

            self._classifier = pipeline(
                "text-classification", model=self.model_name,
                tokenizer=self.model_name, device=-1,
            )
            self.ready = True
        except Exception as exc:
            print(f"Sentiment transformer initialization failed ({exc}); using heuristic fallback.")
            self.backend_name = "heuristic-fallback"
            self.ready = False

    def predict(self, text: str) -> str:
        if self._classifier is None:
            return detect_sentiment(text)
        result = self._classifier(text[:512], truncation=True, max_length=512)[0]
        label = str(result["label"]).lower()
        if "positive" in label or label in {"label_2", "2"}:
            return "positive"
        if "negative" in label or label in {"label_0", "0"}:
            return "negative"
        return "neutral"


def detect_language(text: str) -> str:
    if re.search(r"[\u0600-\u06ff]", text):
        return "ar"
    if re.search(r"[\u0400-\u04ff]", text):
        return "ru"
    if re.search(r"[\u4e00-\u9fff]", text):
        return "zh"
    if re.search(r"[\u3040-\u30ff]", text):
        return "ja"
    if text.isascii():
        return "en"
    try:
        model, vectorizer, encoder = _language_artifacts()
        prediction = model.predict(vectorizer.transform([text]))
        return str(encoder.inverse_transform(prediction)[0])
    except Exception:
        return "en"


def detect_sentiment(text: str) -> str:
    tokens = set(re.findall(r"[a-z']+", text.lower()))
    negative = len(tokens & NEGATIVE_WORDS)
    positive = len(tokens & POSITIVE_WORDS)
    if negative > positive:
        return "negative"
    if positive > negative:
        return "positive"
    return "neutral"


def detect_intent(text: str) -> str:
    lowered = " ".join(text.lower().split())
    for intent, phrases in INTENT_PATTERNS:
        if any(phrase in lowered for phrase in phrases):
            return intent
    try:
        model, vectorizer, encoder = _intent_artifacts()
        prediction = model.predict(vectorizer.transform([text]))
        return str(encoder.inverse_transform(prediction)[0])
    except Exception:
        return "spotify_support"


@lru_cache(maxsize=1)
def _language_artifacts():
    import joblib

    root = Path(__file__).resolve().parents[1] / "models"
    return (
        joblib.load(root / "language_detector.pkl"),
        joblib.load(root / "language_vectorizer.pkl"),
        joblib.load(root / "language_encoder.pkl"),
    )


@lru_cache(maxsize=1)
def _intent_artifacts():
    import joblib

    root = Path(__file__).resolve().parents[1] / "models"
    return (
        joblib.load(root / "intent_model.pkl"),
        joblib.load(root / "intent_vectorizer.pkl"),
        joblib.load(root / "intent_encoder.pkl"),
    )
