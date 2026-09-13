from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import chromadb
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv
load_dotenv()
from groq import Groq

# Initialize FastAPI app
app = FastAPI(title="E-commerce Customer Support Chatbot")

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    language: str
    sentiment: str
    intent: str
    escalated: bool = False

# ==========================================
# 1. Load Models (Mock implementation if models aren't trained yet)
# ==========================================
print("Loading models...")

# 1. Language Detection
try:
    lang_model = joblib.load('models/language_detector.pkl')
    lang_vectorizer = joblib.load('models/language_vectorizer.pkl')
    lang_encoder = joblib.load('models/language_encoder.pkl')
except:
    print("Warning: Language models not found. Ensure Notebook 1 was run.")
    lang_model, lang_vectorizer, lang_encoder = None, None, None

# 2. Sentiment Classifier
try:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sentiment_model = DistilBertForSequenceClassification.from_pretrained('models/sentiment_model').to(device)
    sentiment_tokenizer = DistilBertTokenizer.from_pretrained('models/sentiment_model')
    sentiment_encoder = joblib.load('models/sentiment_encoder.pkl')
except:
    print("Warning: Sentiment models not found. Ensure Notebook 2 was run.")
    sentiment_model, sentiment_tokenizer, sentiment_encoder = None, None, None

# 3. Intent Classifier
try:
    intent_model = joblib.load('models/intent_model.pkl')
    intent_vectorizer = joblib.load('models/intent_vectorizer.pkl')
    intent_encoder = joblib.load('models/intent_encoder.pkl')
except:
    print("Warning: Intent models not found. Ensure Notebook 3 was run.")
    intent_model, intent_vectorizer, intent_encoder = None, None, None

# 4. RAG Pipeline
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
chroma_client = chromadb.PersistentClient(path='./chroma_db')
collection_name = 'customer_support_kb'
try:
    collection = chroma_client.get_collection(name=collection_name)
except:
    print("Warning: ChromaDB collection not found. Ensure Notebook 4 was run.")
    collection = None

# ==========================================
# Helper Functions
# ==========================================
def detect_language(text):
    if not lang_model: return 'en' # fallback
    vec = lang_vectorizer.transform([text])
    pred = lang_model.predict(vec)
    return lang_encoder.inverse_transform(pred)[0]

def detect_sentiment(text):
    if not sentiment_model: return 'neutral' # fallback
    inputs = sentiment_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
    with torch.no_grad():
        logits = sentiment_model(**inputs).logits
    pred = torch.argmax(logits, dim=1).cpu().numpy()
    return sentiment_encoder.inverse_transform(pred)[0]

def detect_intent(text):
    if not intent_model: return 'out_of_scope' # fallback
    vec = intent_vectorizer.transform([text])
    pred = intent_model.predict(vec)
    return intent_encoder.inverse_transform(pred)[0]

def retrieve_context(query, top_k=3):
    if not collection: return [""]
    query_embedding = embedding_model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=top_k)
    return results['documents'][0] if results['documents'] else [""]

def generate_rag_response(user_message, sentiment):
    api_key = os.environ.get('GROQ_API_KEY')
    if not api_key:
        return "I apologize, but my generation system is currently offline (GROQ_API_KEY missing)."
    
    client = Groq(api_key=api_key)
    retrieved_chunks = retrieve_context(user_message)
    context_str = "\\n".join(retrieved_chunks)

    system_prompt = f"""You are a helpful, professional customer support assistant
for an online retailer. Answer the customer's question using ONLY
the information in the retrieved support responses below. If the
customer sounds frustrated ({sentiment}), acknowledge
that before answering. If the retrieved context does not cover
the question, say so honestly and offer to escalate to a human
agent rather than guessing."""

    user_prompt = f"Context:\\n{context_str}\\n\\nCustomer question: \"{user_message}\""

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model="openai/gpt-oss-20b",
            temperature=0.0
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"I apologize, but I encountered an error generating a response: {str(e)}"

# ==========================================
# API Endpoint
# ==========================================
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    return FileResponse("static/index.html")

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    user_msg = request.message

    # 1. Language Detection
    lang = detect_language(user_msg)
    
    # Optional: If language is not English, we could translate it or reply that we only support English
    # But for this scope, we just proceed.
    
    # 2. Sentiment Classification
    sentiment = detect_sentiment(user_msg)
    
    # 3. Intent Classification (with small talk rule-based fallback)
    lower_msg = user_msg.lower()
    if any(word in lower_msg for word in ['hello', 'hi ', 'hey', 'greetings', 'can you help me']):
        intent = 'greeting'
    elif any(word in lower_msg for word in ['thanks', 'thank you', 'appreciate']):
        intent = 'gratitude'
    elif any(word in lower_msg for word in ['bye', 'goodbye']):
        intent = 'goodbye'
    else:
        intent = detect_intent(user_msg)
    
    # 4. Routing and RAG
    escalated = False
    final_response = ""

    # Routing logic based on guidelines
    if intent in ['complaint']:
        # Let's escalate to human directly for high priority complaints
        escalated = True
        final_response = "I am so sorry to hear about your frustration. I am escalating this to a human agent immediately to resolve this for you."
    elif intent in ['greeting', 'goodbye', 'gratitude']:
        # Small talk, no RAG needed
        final_response = "Hello! I am your AI support assistant. How can I help you today?"
    else:
        # Default RAG flow for order_status, billing, etc.
        final_response = generate_rag_response(user_msg, sentiment)

    return ChatResponse(
        response=final_response,
        language=lang,
        sentiment=sentiment,
        intent=intent,
        escalated=escalated
    )

if __name__ == "__main__":
    import uvicorn
    # To run: uvicorn app:app --reload
    uvicorn.run(app, host="localhost", port=8000)
