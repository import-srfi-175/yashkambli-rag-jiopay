---
title: JioPay RAG Chatbot
emoji: 💳
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.25.0
python_version: 3.9
app_file: src/web/streamlit_app.py
---

# JioPay RAG Chatbot

A production-grade Retrieval-Augmented Generation (RAG) chatbot for JioPay customer support automation.

## 🚀 How to Use

This is a Streamlit application that allows you to ask questions about JioPay. The chatbot will use a RAG pipeline to answer your questions based on a knowledge base of JioPay's public documentation.

## 📁 Project Structure

```
yashkambli-rag-jiopay/
├── src/
│   ├── scraping/          # Web scraping modules
│   ├── processing/        # Data cleaning and chunking
│   ├── embeddings/        # Embedding model comparisons
│   ├── retrieval/         # Vector search and retrieval
│   ├── generation/        # LLM generation with citations
│   ├── frontend/          # Web UI components
│   ├── evaluation/        # Evaluation metrics and testing
│   └── main.py           # FastAPI application
├── data/
│   ├── scraped/          # Raw scraped data
│   ├── processed/        # Cleaned and chunked data
│   ├── chroma_db/       # ChromaDB vector store
│   ├── faiss_index/     # FAISS index files
│   └── evaluation/      # Test datasets
├── tests/               # Unit and integration tests
├── reports/             # PDF reports and documentation
├── requirements.txt     # Python dependencies
├── env.example         # Environment variables template
└── README.md           # This file
```