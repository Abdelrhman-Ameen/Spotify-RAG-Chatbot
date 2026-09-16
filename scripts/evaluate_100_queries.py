"""Evaluate 100 independently written, human-style Spotify support queries.

The cases avoid knowledge-base question wording and include shorthand, spelling
mistakes, emotional requests, vague follow-ups, and topic switches.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from spotify_rag.assistant import SpotifyAssistant
from spotify_rag.config import Settings

HUMAN_QUERIES = {
    "refund-policy": ["i want my money back", "can spotify give me a refund pls", "how can i reverse what i paid"],
    "charged-twice": ["you took the monthly fee two times", "there are 2 spotify payments on my bank app", "ive been double billed"],
    "canceled-still-charged": ["i ended premium last week but you billed me again", "why am i paying when i already cancelled", "still got charged after stopping my plan"],
    "unknown-charge": ["what is this spotify debit i never signed up", "spotify took cash but i dont use it", "unrecognized charge from spotify on my card"],
    "failed-payment": ["my card keeps getting declined for premium", "checkout wont accept my visa", "why cant spotify take the monthly fee"],
    "payment-details": ["need to swap the card on my plan", "where can i put my new visa", "use a different payment method next month"],
    "cancel-premium": ["stop renewing my paid plan", "i dont want premium next month", "where do i turn off my membership"],
    "premium-pricing": ["whats the monthly fee where i live", "how expensive is individual premium", "price for the paid spotify plans pls"],
    "subscribe-premium": ["i wanna upgrade from free", "where can i purchase premium", "help me start a paid membership"],
    "premium-benefits": ["what makes paid spotify better", "do i lose ads if i upgrade", "is the premium version actually worth buying"],
    "family-plan": ["can everyone at home share one plan", "how do i add my wife to spotify", "tell me how the household membership works"],
    "duo-plan": ["need a subscription for me and my partner", "is there a two person membership", "how does the couples plan work"],
    "student-plan": ["do university students pay less", "im in college can i get a discount", "how often must i prove student status"],
    "offline-listening": ["can i hear my music on a flight without wifi", "my saved downloads vanished", "what is the offline song limit"],
    "audio-quality": ["where do i make streaming sound better", "change the bitrate in the app", "music quality is low how can i raise it"],
    "lossless": ["can spotify stream flac", "where is the hi res setting", "does bluetooth keep full lossless quality"],
    "spotify-offline-error": ["the app says offline but my wifi works", "spotify thinks i have no internet", "why wont it go back online"],
    "spotify-not-playing": ["songs refuse to start", "every track stops after a second", "i press play and nothing happens"],
    "no-sound": ["the progress bar moves but i hear nothing", "spotify is silent even at full volume", "no audio coming out of the app"],
    "storage": ["spotify is eating all my phone space", "how can i wipe the app cache", "move downloaded songs to my sd card"],
    "data-saver": ["spotify burns too much cellular data", "make the app use fewer megabytes", "what settings save my mobile allowance"],
    "cannot-login": ["i cant remember which email i used", "none of my sign in methods work", "locked out and forgot my username"],
    "reset-password": ["send me a link to pick a new password", "i need to change my secret login", "where is the password reset"],
    "account-hacked": ["some stranger keeps changing my songs", "my spotify was taken over", "kick every unknown device off my account"],
    "delete-account": ["erase my spotify profile permanently", "i want all my account data gone", "how can i close spotify for good"],
    "recover-playlist": ["oops i removed a playlist can i undo it", "bring back a playlist i deleted yesterday", "my list is gone how do i restore it"],
    "collaborative-playlist": ["let my friends put tracks in my list", "how can several people edit one playlist", "invite someone to add music with me"],
    "spotify-connect": ["send the music from my phone to the tv", "my speaker doesnt appear in available devices", "control desktop playback using mobile"],
    "private-listening": ["hide what im listening to for a while", "i dont want followers seeing todays music", "turn on a secret listening session"],
    "privacy-data": ["download a copy of everything spotify knows about me", "what personal info are you collecting", "where are my privacy choices"],
}

BEHAVIOR_CASES = [
    {"message": "yo there", "intent": "greeting"},
    {"message": "thx that fixed it", "intent": "gratitude"},
    {"message": "ok im done cya", "intent": "goodbye"},
    {"message": "honestly i really love this app", "intent": "opinion", "sentiment": "positive"},
    {"message": "spotify is the worst app ive paid for", "intent": "opinion", "sentiment": "negative"},
    {"message": "no i want a refund", "source": "refund-policy"},
    {"message": "how do i do that", "history": [{"role": "user", "content": "I need to stop renewing my Premium plan"}], "source": "cancel-premium"},
    {"message": "does it work without wifi though", "history": [{"role": "user", "content": "I want to download songs before my flight"}], "source": "offline-listening"},
    {"message": "what about for two people", "history": [{"role": "user", "content": "Which Spotify Premium plan should I get?"}], "source": "duo-plan"},
    {"message": "i was asking about my money, give it back", "source": "refund-policy"},
]


def main() -> int:
    knowledge = json.loads((ROOT / "data" / "spotify_knowledge_base.json").read_text(encoding="utf-8"))
    titles = {entry["id"]: entry["title"] for entry in knowledge["entries"]}
    cases = [
        {"message": message, "source": entry_id}
        for entry_id, messages in HUMAN_QUERIES.items()
        for message in messages
    ] + BEHAVIOR_CASES
    if len(cases) != 100:
        raise SystemExit(f"Evaluation must contain exactly 100 cases; found {len(cases)}")

    settings = replace(
        Settings.from_env(ROOT), retrieval_backend="tfidf", llm_backend="extractive",
        sentiment_backend="heuristic", relevance_threshold=0.20,
    )
    assistant = SpotifyAssistant(settings)
    assistant.initialize()
    failures = []
    for index, case in enumerate(cases, start=1):
        result = assistant.chat(case["message"], history=case.get("history", []))
        reasons = []
        if source_id := case.get("source"):
            expected = titles[source_id]
            actual = result["sources"][0]["title"] if result["sources"] else None
            if actual != expected:
                reasons.append(f"source={actual!r}, expected={expected!r}")
        if expected := case.get("intent"):
            if result["intent"] != expected:
                reasons.append(f"intent={result['intent']!r}, expected={expected!r}")
        if expected := case.get("sentiment"):
            if result["sentiment"] != expected:
                reasons.append(f"sentiment={result['sentiment']!r}, expected={expected!r}")
        if reasons:
            failures.append(f"{index:03d} {case['message']!r}: {'; '.join(reasons)}")

    passed = len(cases) - len(failures)
    print(f"Independent human-query evaluation: {passed}/{len(cases)} passed")
    if failures:
        print("\n".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
