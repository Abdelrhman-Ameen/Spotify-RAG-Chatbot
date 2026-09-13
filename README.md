# ITI Chatbot Project

This project is a Retrieval-Augmented Generation (RAG) based customer support chatbot for an e-commerce platform. It integrates multiple NLP modules to provide accurate, tone-appropriate, and grounded responses.

## Features
1. **Language Detection**: Automatically detects the language of the customer's message.
2. **Sentiment & Emotion Classification**: Identifies whether the customer sounds frustrated, neutral, or satisfied.
3. **Intent Classification**: Routes the message to the correct handling path.
4. **Q&A RAG Pipeline**: Retrieves grounded information from a vector database using sentence-transformers and Groq LLMs.

## Project Structure
- `1_language_detection.ipynb`: Notebook for training the language detection module.
- `2_sentiment_classifier.ipynb`: Notebook for training the sentiment classification module.
- `3_intent_classifier.ipynb`: Notebook for training the intent classification module.
- `4_rag_pipeline.ipynb`: Notebook for building the RAG pipeline and vector database.
- `app.py`: FastAPI backend that serves the chatbot.
- `static/`: Frontend interface for the chatbot.
- `models/`: Directory containing the saved machine learning models.
- `requirements.txt`: Python dependencies required to run the project.

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/MariamElbadry/ITI_Chatbot.git
   cd ITI_Chatbot
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Environment Variables:**
   Create a `.env` file in the root directory and add your Groq API key:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

4. **Run the Notebooks:**
   Before running the app, ensure you have run the four Jupyter Notebooks in order (1 to 4) to train the models and build the Chroma vector database locally.

5. **Run the Application:**
   ```bash
   uvicorn app:app --reload
   ```

6. **Access the Chatbot:**
   Open your browser and navigate to `http://localhost:8000` to interact with the chatbot.
