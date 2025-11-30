# RAG: Multi-Document Q&A App

This project implements a **Retrieval-Augmented Generation (RAG)** system with a Streamlit frontend and modular backend, enabling users to upload a variety of document types (PDF, TXT, DOCX, PPTX, Excel, CSV) and ask questions over their content.
It uses LangChain + ChromaDB + OpenAI for embeddings & LLM, and integrates LangSmith for tracing and observability of your pipeline.

---

## 📂 File Structure

```
rag-streamlit/
├─ .env.example
├─ README.md
├─ requirements.txt
├─ .streamlit/
│  └─ config.toml
├─ chroma_db/                # Persistent vector store (Chroma)
├─ app.py                    # Streamlit UI entrypoint
├─ src/
│  ├─ __init__.py
│  ├─ config.py              # env + settings handling (including LangSmith)
│  ├─ ingest.py              # ingestion & loaders (PDF, TXT, DOCX, Excel, CSV)
│  ├─ vectorstore.py         # ChromaDB wrapper logic
│  ├─ chains.py              # retrieval + QA chain logic
│  └─ utils.py               # helpers (file handling, text splitting, etc.)
└─ notebooks/
   └─ quick_test.ipynb       # optional exploratory notebook
```

---



## 🔍 System Architecture & Flow

```
┌───────────────┐
│   .env file   │ → defines OPENAI, LANGSMITH and other settings
└──────┬────────┘
       │
       ▼
┌─────────────────────┐
│  Settings (config)  │ → loads envs, sets defaults for models, tracing
└──────┬──────────────┘
       │
       ▼
┌───────────────────────────────┐
│   Streamlit UI (app.py)       │ → upload docs, ingestion trigger, ask queries
└──────────────┬────────────────┘
               │ uploads docs
               ▼
┌─────────────────────────────────────┐
│   Ingestion (ingest.py)             │ → loads PDF/TXT/DOCX/Excel/CSV, splits text
└──────────────┬──────────────────────┘
               │ docs→chunks
               ▼
┌────────────────────────────────────────┐
│   Vector Store (vectorstore.py)        │ → embed chunks, store/retrieve via ChromaDB
└──────────────┬─────────────────────────┘
               │ retrieval
               ▼
┌────────────────────────────────────────┐
│   QA Chain (chains.py)                │ → uses retriever + OpenAI LLM to answer
└──────────────┬─────────────────────────┘
               │ answers
               ▼
┌─────────────────────────────────────────┐
│   Streamlit Output (app.py)            │ → show answer + source snippets
└─────────────────────────────────────────┘
```

---

## Key Features

* ✅ Multi-file ingestion: PDF, TXT, MD, DOCX, PPTX, Excel (.xlsx/.xls) and CSV.
* ✅ Persistent vector store via ChromaDB (`chroma_db/`) for reuse of embeddings across sessions.
* ✅ Natural language Q&A powered by OpenAI LLMs.
* ✅ Full observability: integrated with LangSmith to trace ingestion, embedding, retrieval & LLM steps.
* ✅ Modular architecture: separated config, ingestion, vectorstore, chain, UI for maintainability & extensibility.
* ✅ Adjustable parameters: chunk size, overlap, model names, and tracing toggles via environment.

---

## LangSmith Observability

* You can enable tracing by setting `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY` environment variables. ([docs.smith.langchain.com][1])
* Use the `@traceable` decorator or context manager from the LangSmith SDK to trace functions, chains or entire pipelines. ([docs.smith.langchain.com][2])
* In the LangSmith UI you’ll see spans for each step: ingestion, embedding, retrieval, LLM call — grouped under a trace representing a user query. ([docs.smith.langchain.com][3])

---

## 🧩 Module Overview

| Module           | Description                                                                  |
| ---------------- | ---------------------------------------------------------------------------- |
| `config.py`      | Loads `.env`, sets up models, vector store path, LangSmith tracing config.   |
| `ingest.py`      | Document loaders + chunking logic (including Excel/CSV support).             |
| `vectorstore.py` | Handles embedding generation + persistent storage & similarity search.       |
| `chains.py`      | Builds the retrieval + LLM QA chain pipeline.                                |
| `utils.py`       | Helper utilities (file handling, chunking splitter, format detection).       |
| `app.py`         | Streamlit UI: file upload, ingestion, query input, answer & sources display. |
