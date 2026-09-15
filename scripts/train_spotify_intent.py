"""Train the compact Spotify support intent classifier from curated KB questions."""

import json
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
entries = json.loads((ROOT / "data" / "spotify_knowledge_base.json").read_text(encoding="utf-8"))["entries"]
texts = [question for entry in entries for question in entry["questions"]]
labels = [entry["category"] for entry in entries for _ in entry["questions"]]

encoder = LabelEncoder()
y = encoder.fit_transform(labels)
vectorizer = TfidfVectorizer(ngram_range=(1, 3), analyzer="char_wb", min_df=1)
X = vectorizer.fit_transform(texts)
model = LinearSVC(random_state=42).fit(X, y)

models = ROOT / "models"
models.mkdir(exist_ok=True)
joblib.dump(model, models / "intent_model.pkl")
joblib.dump(vectorizer, models / "intent_vectorizer.pkl")
joblib.dump(encoder, models / "intent_encoder.pkl")
print(f"Saved {len(encoder.classes_)} Spotify intents trained on {len(texts)} questions.")
