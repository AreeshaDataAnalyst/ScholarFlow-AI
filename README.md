# ScholarFlow-AI
AI-powered PDF study assistant using RAG, FAISS, Sentence Transformers, Streamlit, and Groq.
# 📚 ScholarFlow AI

### Your PDF. Your Questions. Your Study Assistant.

ScholarFlow AI is an AI-powered PDF study assistant built using **Python, Streamlit, FAISS, Sentence Transformers, and Groq**.

It allows students to upload a text-based PDF, ask questions about their study material, and receive answers based on the content of their uploaded document.

---

## ✨ Features

- 📄 PDF-only upload
- 🔎 Semantic search using FAISS
- 🧠 Sentence Transformer embeddings
- 💬 Natural-language question answering
- 🤖 Groq-powered AI responses
- 📖 Source page references
- 🚫 Rejects scanned and image-only PDFs
- 🔐 Secure API key management using Streamlit Secrets
- 🎓 Student-friendly answers
- ⚡ Fast document retrieval
- 🎨 Clean and modern Streamlit interface

---

## 🧠 How It Works

ScholarFlow AI uses a **Retrieval-Augmented Generation (RAG)** approach.

```text
Upload PDF
    ↓
Extract Text
    ↓
Create Text Chunks
    ↓
Generate Embeddings
    ↓
Store Embeddings in FAISS
    ↓
User Asks a Question
    ↓
Search Relevant PDF Content
    ↓
Send Retrieved Context to AI
    ↓
Generate Answer
    ↓
Display Answer + Source Pages

ScholarFlow-AI/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
└── .streamlit/
    └── secrets.toml

| Technology            | Purpose                 |
| --------------------- | ----------------------- |
| Python                | Application development |
| Streamlit             | Web application         |
| PyMuPDF               | PDF text extraction     |
| Sentence Transformers | Text embeddings         |
| FAISS                 | Semantic search         |
| Groq                  | AI text generation      |
| NumPy                 | Numerical processing    |
