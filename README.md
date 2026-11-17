# DocuQuery – AI-Powered PDF Question Answering

DocuQuery is an AI-powered document question-answering application built using **Retrieval-Augmented Generation (RAG)**. It allows users to ask questions about PDF documents and receive relevant, context-based answers using Large Language Models (LLMs).

The project is designed to explore and implement modern AI application development using LangChain, Gemini, vector databases, and LangSmith.

## 🎯 Project Objectives

- Upload and process PDF documents.
- Extract text and split it into smaller chunks.
- Generate embeddings for document chunks.
- Store embeddings in a vector database.
- Retrieve relevant information based on user questions.
- Generate context-aware answers using Gemini.
- Monitor and evaluate application performance using LangSmith.

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Backend programming |
| LangChain | RAG application development |
| Google Gemini | LLM and embeddings |
| ChromaDB | Vector storage and similarity search |
| Qdrant | Alternative vector database |
| LangSmith | Tracing, monitoring, and evaluation |
| FastAPI | Backend API (planned) |
| Git & GitHub | Version control and collaboration |

## 🔄 How It Works

1. **Document Upload:** User provides a PDF document.
2. **Text Extraction:** Extract text from the PDF.
3. **Text Splitting:** Divide extracted text into smaller chunks.
4. **Embedding Generation:** Convert chunks into numerical vectors.
5. **Vector Storage:** Store embeddings in ChromaDB.
6. **Retrieval:** Find relevant document chunks for a question.
7. **Answer Generation:** Gemini generates an answer using retrieved context.
8. **Monitoring:** LangSmith helps trace and evaluate the RAG pipeline.

## 🏗️ Project Architecture

```text
             PDF Document
                   |
                   v
            PDF Text Loader
                   |
                   v
              Text Splitter
                   |
                   v
           Gemini Embeddings
                   |
                   v
              ChromaDB
                   |
                   v
User Question --> Retriever
                   |
                   v
          Relevant Documents
                   |
                   v
             Gemini LLM
                   |
                   v
             Final Answer
                   |
                   v
              LangSmith
          (Tracing & Evaluation)
```

## 📁 Project Structure

```text
DocuQuery/
│
├── app/
│   ├── __init__.py
│   ├── config.py
│   ├── document_loader.py
│   ├── text_splitter.py
│   ├── embeddings.py
│   ├── vector_store.py
│   └── main.py
│
├── data/
│   └── sample.pdf
│
├── chroma_db/
│
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```

*Note: The structure represents the planned application structure and will grow as development progresses.*

## 🚀 Development Roadmap

### Stage 1 – Basic RAG Application
- [ ] Set up Python environment.
- [ ] Load PDF documents.
- [ ] Extract and split document text.
- [ ] Generate embeddings using Gemini.
- [ ] Store vectors in ChromaDB.
- [ ] Implement similarity search.
- [ ] Generate answers using Gemini.
- [ ] Build a basic RAG pipeline.

### Stage 2 – Advanced RAG Application
- [ ] Support multiple PDF uploads.
- [ ] Add conversational question answering.
- [ ] Display source documents and page numbers.
- [ ] Implement metadata filtering.
- [ ] Integrate Qdrant vector database.
- [ ] Add document management.

### Stage 3 – Monitoring and Deployment
- [ ] Integrate LangSmith tracing.
- [ ] Evaluate RAG responses.
- [ ] Build FastAPI endpoints.
- [ ] Develop a user-friendly frontend.
- [ ] Add Docker support.
- [ ] Prepare the application for deployment.

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/DocuQuery.git
cd DocuQuery
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the environment

**Windows:**

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure environment variables

Create a `.env` file in the project root and add your API configuration:

```env
GEMINI_API_KEY=your_gemini_api_key

LANGSMITH_TRACING=false
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=DocuQuery
```

Keep your API keys private. Never commit your `.env` file to GitHub.

## 📚 Learning Outcomes

Through this project, I aim to understand:

- Retrieval-Augmented Generation (RAG).
- LangChain components and document processing.
- Embeddings and semantic search.
- Vector databases such as ChromaDB and Qdrant.
- LLM integration using Google Gemini.
- LangSmith tracing and evaluation.
- Building and deploying AI-powered applications.

## 👩‍💻 Project Status

**Currently in development – Stage 1: Basic RAG Application.**

This is a learning and practical implementation project focused on understanding how production-oriented RAG applications are designed and developed.

## 📌 Future Scope

The application will gradually evolve into a complete document intelligence platform with advanced retrieval, conversational memory, source citations, evaluation, and deployment capabilities.

---

**Built with Python, LangChain, Gemini, Vector Databases, and LangSmith.**