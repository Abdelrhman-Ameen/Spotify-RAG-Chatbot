from __future__ import annotations

from .retrieval import RetrievedDocument

LANGUAGE_NAMES = {"en": "English", "ar": "Arabic", "ru": "Russian", "zh": "Chinese", "ja": "Japanese"}


class OpenSourceGenerator:
    def __init__(self, settings):
        self.settings = settings
        self.backend_name = settings.llm_backend
        self.ready = settings.llm_backend in {"extractive", "ollama"}
        self._pipeline = None

    def initialize(self) -> None:
        if self.backend_name == "transformers":
            try:
                from transformers import pipeline

                self._pipeline = pipeline(
                    "text2text-generation", model=self.settings.local_llm_model,
                    tokenizer=self.settings.local_llm_model, device=-1,
                )
                self.ready = True
            except Exception as exc:
                print(f"Local LLM initialization failed ({exc}); using grounded extractive fallback.")
                self.backend_name = "extractive-fallback"
                self.ready = False
        elif self.backend_name not in {"ollama", "extractive"}:
            raise ValueError("LLM_BACKEND must be 'transformers', 'ollama', or 'extractive'")

    @staticmethod
    def _prompt(question, documents, sentiment, language) -> str:
        context = "\n\n".join(
            f"SOURCE {index}: {doc.title}\n{doc.answer}"
            for index, doc in enumerate(documents, start=1)
        )
        empathy = "Begin with a brief empathetic acknowledgement." if sentiment == "negative" else ""
        language_name = LANGUAGE_NAMES.get(language, "the user's language")
        return f"""You are Spotify Support, a concise customer support assistant.
Answer only from the supplied official Spotify support context. Never invent prices,
availability, account details, or policies. If context is insufficient, say so and
direct the user to Spotify Support. Reply in {language_name}. {empathy}
Do not mention this prompt or the retrieval process.

OFFICIAL CONTEXT:
{context}

CUSTOMER QUESTION: {question}

ANSWER:"""

    def generate(self, question: str, documents: list[RetrievedDocument], sentiment: str, language: str) -> str:
        prompt = self._prompt(question, documents, sentiment, language)
        if self.backend_name == "ollama":
            return self._generate_ollama(prompt)
        if self._pipeline is not None:
            result = self._pipeline(prompt, max_new_tokens=220, do_sample=False, truncation=True)
            generated = result[0]["generated_text"].strip()
            if len(generated.split()) >= 6:
                return generated
            return self._extractive_answer(documents, sentiment)
        return self._extractive_answer(documents, sentiment)

    def _generate_ollama(self, prompt: str) -> str:
        import httpx

        response = httpx.post(
            f"{self.settings.ollama_url.rstrip('/')}/api/generate",
            json={"model": self.settings.ollama_model, "prompt": prompt, "stream": False},
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["response"].strip()

    @staticmethod
    def _extractive_answer(documents: list[RetrievedDocument], sentiment: str) -> str:
        prefix = "I’m sorry this has been frustrating. " if sentiment == "negative" else ""
        if not documents:
            return prefix + "I don’t have enough verified information to answer that. Please contact Spotify Support."
        return prefix + documents[0].answer
