from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

NEGATIVE_WORDS = {
    "angry", "annoyed", "awful", "bad", "broken", "charged", "complaint",
    "disappointed", "frustrated", "hate", "horrible", "refund", "reimburse", "scam",
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
    ("greeting", ("good morning", "good evening")),
    ("gratitude", ("thank you", "thanks", "thx", "appreciate")),
    ("goodbye", ("goodbye", "bye", "cya", "see you", "im done")),
    ("opinion", OPINION_PHRASES),
    ("opinion", ("worst app", "best app", "love this", "hate this")),
    ("account_security", ("hacked", "someone accessed", "stolen account", "sign out everywhere", "taken over", "stranger", "unknown device", "kick every")),
    ("refund_request", ("refund", "money back", "cash back", "reimburse", "reverse what i paid", "reverse the charge", "return my payment", "give it back")),
    ("duplicate_charge", ("charged twice", "charged two times", "double charged", "double billed", "duplicate charge", "two charges", "2 spotify payments", "fee two times", "multiple charges")),
    ("canceled_but_charged", ("canceled but charged", "cancelled but charged", "charged after cancel", "charged after stopping", "billed me again", "already cancelled", "still being charged", "still got charged")),
    ("unknown_charge", ("unrecognized charge", "unknown charge", "never signed up", "dont use it", "don't use it", "didnt buy", "spotify debit")),
    ("payment_details", ("swap the card", "new visa", "new card", "different payment method", "change payment", "update my card", "payment details")),
    ("failed_payment", ("charge my card", "payment fail", "payment failed", "failed payment", "cannot charge", "cant spotify take", "card keeps getting declined", "card declined", "checkout wont accept", "checkout won't accept")),
    ("cancel_premium", ("cancel plan", "cancel premium", "cancel subscription", "stop renewing", "dont want premium", "don't want premium", "turn off my membership", "end premium", "stop my plan")),
    ("premium_pricing", ("price of spotify", "spotify price", "how much is spotify", "how much does spotify", "premium cost", "cost of spotify", "price of premium", "premium price", "monthly fee", "how expensive", "price for the paid")),
    ("premium_benefits", ("why buy spotify", "why do i buy spotify", "why would i buy spotify", "why should i buy spotify", "why get premium", "why should i get premium", "is premium worth", "premium benefits", "benefits of premium", "what do i get with premium", "paid spotify better", "lose ads", "worth buying")),
    ("premium_subscription", ("how to subscribe", "how do i subscribe", "get spotify premium", "sign up for premium", "buy a spotify subscription", "upgrade from free", "purchase premium", "start a paid membership")),
    ("family_plan", ("family plan", "everyone at home", "household", "add my wife", "add my husband", "add family")),
    ("duo_plan", ("duo", "two person", "two people", "couples plan", "me and my partner", "for my partner")),
    ("student_plan", ("student", "university", "college", "campus discount")),
    ("login_help", ("cannot sign in", "can't sign in", "cannot log in", "can't log in", "forgot my login", "forgot my spotify login", "forgot my email", "forgot my username")),
    ("login_help", ("cant remember which email", "can't remember which email", "sign in methods", "locked out", "forgot my username")),
    ("reset_password", ("password reset", "new password", "change my password", "reset my password", "secret login")),
    ("delete_account", ("delete account", "close spotify", "close account", "profile permanently", "account data gone", "erase my spotify")),
    ("account_management", ("password", "log in", "login", "sign in", "sign up", "email address", "username")),
    ("billing_and_refunds", ("charged", "payment", "billing", "invoice", "receipt", "gift card")),
    ("basic_plan", ("spotify basic", "basic plan", "basic individual", "basic family", "basic duo")),
    ("data_usage", ("save data", "use less data", "data saver", "mobile data", "reduce data")),
    ("data_usage", ("cellular data", "fewer megabytes", "mobile allowance")),
    ("offline_listening", ("without wifi", "offline listening", "downloads vanished", "offline song limit", "download songs", "saved downloads")),
    ("lossless_audio", ("lossless", "flac", "hi res", "hi-res")),
    ("audio_quality", ("audio quality", "sound better", "bitrate", "streaming quality", "music quality")),
    ("connection_issue", ("says offline", "thinks i have no internet", "go back online", "wifi works")),
    ("no_sound", ("no sound", "hear nothing", "silent", "no audio", "full volume")),
    ("playback_issue", ("not playing", "wont play", "won't play", "refuse to start", "track stops", "press play", "keeps stopping")),
    ("storage_help", ("storage", "phone space", "app cache", "clear cache", "sd card")),
    ("playlist_recovery", ("recover playlist", "restore playlist", "playlist i deleted", "removed a playlist", "playlist is gone", "list is gone")),
    ("collaborative_playlist", ("collaborative", "friends put tracks", "people edit one playlist", "add music with me", "friends add songs")),
    ("spotify_connect", ("spotify connect", "phone to the tv", "phone to tv", "available devices", "control desktop", "play on a speaker")),
    ("private_listening", ("private session", "secret listening", "hide what im listening", "hide what i'm listening", "followers seeing")),
    ("privacy_data", ("personal info", "data spotify knows", "copy of everything", "privacy choices", "request my data")),
    ("playlists_and_library", ("playlist", "library", "liked songs")),
    ("devices", ("connect", "speaker", "tv", "watch", "car", "bluetooth", "device")),
    ("privacy", ("privacy", "data", "listening activity", "public playlist")),
    ("premium_plans", ("premium", "subscription")),
    ("artists_and_creators", ("artist", "podcast", "creator", "upload music", "show")),
    ("audiobooks", ("audiobook", "listening hours", "top-up")),
    ("contact_support", ("contact spotify", "phone number", "talk to a person", "customer service", "contact support")),
    ("spotify_basics", ("what is spotify", "what can i do with spotify", "tell me about spotify", "basic information about spotify")),
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


def detect_explicit_intent(text: str) -> str | None:
    lowered = " ".join(text.lower().split())
    if re.fullmatch(r"(?:hi|hello|hey|yo)(?: there)?[!.?]*", lowered):
        return "greeting"
    for intent, phrases in INTENT_PATTERNS:
        if any(_contains_phrase(lowered, phrase) for phrase in phrases):
            return intent
    return None


def detect_intent(text: str) -> str:
    if explicit := detect_explicit_intent(text):
        return explicit
    try:
        model, vectorizer, encoder = _intent_artifacts()
        prediction = model.predict(vectorizer.transform([text]))
        return str(encoder.inverse_transform(prediction)[0])
    except Exception:
        return "spotify_support"


def _contains_phrase(text: str, phrase: str) -> bool:
    if " " in phrase:
        return phrase in text
    return re.search(rf"\b{re.escape(phrase)}\b", text) is not None


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
