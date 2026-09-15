"""Regenerate the Spotify-specific intent and RAG assignment notebooks."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def cell(kind, source):
    return {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True), **({"outputs": [], "execution_count": None} if kind == "code" else {})}


def write_notebook(name, cells):
    payload = {
        "cells": cells,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.10"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (ROOT / name).write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


write_notebook("3_intent_classifier.ipynb", [
    cell("markdown", "# Spotify Intent Classifier\n\nTrain a compact, explainable classifier from the same Spotify support questions used by the RAG knowledge base."),
    cell("code", "from pathlib import Path\nimport json, joblib\nimport pandas as pd\nfrom sklearn.feature_extraction.text import TfidfVectorizer\nfrom sklearn.preprocessing import LabelEncoder\nfrom sklearn.svm import LinearSVC\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.metrics import classification_report, accuracy_score"),
    cell("markdown", "## Build labeled examples\n\nEvery curated knowledge entry supplies realistic example questions and a routing category. This keeps the intent taxonomy aligned with the deployed Spotify assistant."),
    cell("code", "kb = json.loads(Path('data/spotify_knowledge_base.json').read_text(encoding='utf-8'))\nrows = [\n    {'text': question, 'intent': entry['category']}\n    for entry in kb['entries']\n    for question in entry['questions']\n]\ndf = pd.DataFrame(rows)\nprint(df['intent'].value_counts())\ndf.head()"),
    cell("markdown", "## Train and evaluate\n\nWord and character n-grams handle short support questions and variations such as `login`/`log in`. The deployment also has deterministic rules for high-risk security and complaint routes."),
    cell("code", "encoder = LabelEncoder()\ny = encoder.fit_transform(df['intent'])\nX_train, X_test, y_train, y_test = train_test_split(df['text'], y, test_size=0.25, random_state=42, stratify=y)\nvectorizer = TfidfVectorizer(ngram_range=(1, 2), analyzer='char_wb', min_df=1)\nX_train_vec = vectorizer.fit_transform(X_train)\nX_test_vec = vectorizer.transform(X_test)\nmodel = LinearSVC(random_state=42).fit(X_train_vec, y_train)\npred = model.predict(X_test_vec)\nprint('Accuracy:', accuracy_score(y_test, pred))\nprint(classification_report(y_test, pred, target_names=encoder.classes_, zero_division=0))"),
    cell("markdown", "## Fit all curated examples and save artifacts"),
    cell("code", "X_all = vectorizer.fit_transform(df['text'])\nmodel.fit(X_all, y)\nPath('models').mkdir(exist_ok=True)\njoblib.dump(model, 'models/intent_model.pkl')\njoblib.dump(vectorizer, 'models/intent_vectorizer.pkl')\njoblib.dump(encoder, 'models/intent_encoder.pkl')\nprint('Saved Spotify intent artifacts.')"),
])

write_notebook("4_rag_pipeline.ipynb", [
    cell("markdown", "# Spotify Q&A RAG Pipeline\n\nInspect the production retrieval and open-source generation pipeline used by FastAPI. No paid API key is required."),
    cell("code", "# Run once if dependencies are missing:\n# %pip install -r requirements.txt\nfrom pathlib import Path\nfrom spotify_rag.config import Settings\nfrom spotify_rag.assistant import SpotifyAssistant"),
    cell("markdown", "## Initialize Chroma, multilingual embeddings, and the local LLM\n\nThe first execution downloads `paraphrase-multilingual-MiniLM-L12-v2` and `google/flan-t5-small`, then persists the Spotify vector index in `chroma_db/`."),
    cell("code", "settings = Settings.from_env(Path.cwd())\nassistant = SpotifyAssistant(settings)\nassistant.initialize()\nprint({\n    'knowledge_entries': assistant.retriever.count,\n    'retrieval_backend': assistant.retriever.backend_name,\n    'llm_backend': assistant.generator.backend_name,\n    'llm_ready': assistant.generator.ready,\n})"),
    cell("markdown", "## Inspect semantic retrieval"),
    cell("code", "query = 'How many songs can I save before a flight?'\nfor item in assistant.retriever.search(query, top_k=4):\n    print(f'{item.relevance:.3f}  {item.title}  {item.url}')"),
    cell("markdown", "## Run the integrated pipeline\n\nThe result includes the generated answer, NLP routing metadata, an escalation flag, grounding status, and official source citations."),
    cell("code", "result = assistant.chat(query, top_k=4)\nresult"),
    cell("markdown", "## Hallucination check\n\nAn unrelated question should fall below the relevance threshold and return a safe refusal instead of sending weak context to the LLM."),
    cell("code", "assistant.chat('What is the tensile strength of lunar concrete?')"),
])

print("Updated 3_intent_classifier.ipynb and 4_rag_pipeline.ipynb")
