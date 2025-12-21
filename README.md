# 🤖 RAG Chatbot (Retrieval-Augmented Generation)

A powerful, efficient, and privacy-focused RAG chatbot system built with **FastAPI**, **LangChain**, **ChromaDB**, and **Groq**. This project allows you to chat with your own documents (PDFs, Text files) using state-of-the-art LLMs, and includes a **Visual Learning** module to understand the RAG concepts.

## 🌟 Why RAG?
Standard AI models are trained on general internet data and don't know your private documents. RAG solves this by:
- **🔒 Privacy**: Chat with your private data without training a public model.
- **✅ Accuracy**: Reduces hallucinations by grounding answers in your actual documents.
- **🔄 Up-to-Date**: No need for expensive retraining; just upload new files.
- **💰 Cost Effective**: Uses efficient retrieval instead of massive context windows.

## 🏗️ Architecture
The system follows a modern RAG pipeline:
1.  **Ingest (Frontend/API)**: User uploads a file (PDF/TXT).
2.  **LangChain**: Splits the document into manageable chunks.
3.  **ChromaDB**: Embeds and stores these chunks as vectors for semantic search.
4.  **Query**: User asks a question.
5.  **Retrieve**: The system finds the most relevant chunks from the database.
6.  **Groq API**: The LLM (Llama-3 class) generates a precise answer using the retrieved context.

## 🛠️ Components

### 1. LangChain 🦜🔗
*The Data Splitter*
Handles the loading of documents (PDF, TxT) and splitting them into small, manageable chunks to fit within the AI's context window.

### 2. ChromaDB 🧬
*The Semantic Search Engine*
A vector database that stores text by *meaning* (embeddings), allowing the system to find relevant information even if the exact keywords don't match.

### 3. FastAPI ⚡
*The Data Doorway*
Provides a high-performance API with endpoints for uploading documents (`/upload`) and querying (`/query`).

### 4. Groq API ⚡
*The Speed Engine*
Powers the inference using Llama-3 models, providing near-instant responses.

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- A [Groq API Key](https://console.groq.com/)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Abrash-Official/Rag_Chatbot.git
   cd Rag_Chatbot
   ```

2. **Create and Activate Virtual Environment**
   ```bash
   python -m venv venv
   # Windows
   .\venv\Scripts\activate
   # Mac/Linux
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables**
   Create a `.env` file in the root directory:
   ```ini
   GROQ_API_KEY=your_groq_api_key_here
   # Optional overrides:
   # GROQ_MODEL=llama-3.3-70b-versatile
   # CHROMA_DIR=./chroma_db
   ```

### Usage

1. **Start the API Server**
   ```bash
   uvicorn main:app --reload
   ```

2. **Access the Web Interface**
   Open your browser and navigate to:
   `http://127.0.0.1:8000/form`

3. **Upload & Chat**
   - Use the interface to upload a PDF or Text file.
   - Wait for the "Ingested" status.
   - Ask questions about your document!

### 🎓 Visual Learning
This repository includes an interactive slide deck to explain RAG concepts.
- Open `Visual_Learning/index.html` in your web browser to view the presentation.

## 🧠 Prompts Logic
The system uses sophisticated prompting strategies to ensure quality:
- **System Prompt**: Defines the persona (precise, pragmatic assistant).
- **Refine Template**: Rewrites user queries to be optimal for database retrieval.
- **Developer Prompt**: Instructs the model to stick strictly to the retrieved context.
- **Answer Template**: Combines the refined query and context chunks for the final response.

## 📂 Project Structure
- `main.py`: The core FastAPI application.
- `Visual_Learning/`: Contains the education slide deck (`index.html`).
- `data/`: Directory where uploaded files are stored.
- `chroma_db/`: Directory for persistent vector storage.
- `form.html`: Simple frontend for testing the API.

---
*Built for AI Education.*
