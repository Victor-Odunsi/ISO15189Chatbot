# ISO 15189:2022 RAG Chatbot – Laboratory Quality Management Assistant

## Overview

As a **medical laboratory scientist**, I recognize how critical **Quality Management Systems (QMS)** are for ensuring accurate results, patient safety, and accreditation. However, navigating the **ISO 15189:2022** standard can be challenging for professionals seeking practical guidance.

This project uses **Retrieval-Augmented Generation (RAG)** to build a chatbot that provides **clause-specific, cited answers** from the ISO 15189:2022 document. Users can ask questions such as:
> *"What does ISO 15189 say about equipment calibration?"*

...or ask for the same content as an audit checklist or a Standard Operating Procedure draft. The chatbot retrieves the relevant clauses, generates a grounded response, and cites the source document and page for every answer.

---

## Features

- Hybrid retrieval: dense (Chroma) + sparse (BM25) search, merged with Reciprocal Rank Fusion and reranked with a local cross-encoder
- Query rewriting and intent classification (general question / checklist / SOP) via a single structured-output LLM call
- Real token-by-token streaming (Server-Sent Events), with source citations delivered before the answer streams in
- Persistent, multi-session chat history (Postgres), with a Next.js UI for browsing and resuming past conversations
- Admin endpoint for incrementally ingesting new source documents, protected by a shared secret

---

## Architecture

```
frontend/   Next.js (App Router, TypeScript, Tailwind) — chat UI, SSE client
backend/    FastAPI
  app/api/       routes (chat, admin, sessions)
  app/services/  hybrid retrieval, ingestion, the LCEL chain, LLM client
  app/db/        Postgres persistence (SQLAlchemy + Alembic)
  app/core/      config, security
```

- **LLM**: Groq (`llama-3.1-8b-instant`), with Mistral as a fallback
- **Embeddings**: `BAAI/bge-small-en` (HuggingFace, local/CPU)
- **Reranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2` (local/CPU)
- **Vector store**: Chroma (persisted to disk)
- **Chat history**: Postgres

---

## Getting Started

### Prerequisites

- Python 3.11+, Node.js 20+
- A [Groq](https://console.groq.com) API key (and optionally a [Mistral](https://mistral.ai) key as fallback)
- Docker, if running via `docker-compose` (recommended)
- Your own copy of the ISO 15189:2022 standard as a PDF — it's a licensed document and isn't included in this repo

### Running with Docker Compose

```bash
cp .env.example .env
# fill in GROQ_API_KEY / MISTRALAI_API_KEY in .env

cp your-iso-15189.pdf data/

docker compose up --build
# backend:  http://localhost:8000
# frontend: http://localhost:3000
```

The backend runs Alembic migrations automatically on startup. Once the stack is up, ingest your source document:

```bash
curl -X POST http://localhost:8000/admin/upload-doc/ \
  -H "X-Admin-Key: $ADMIN_API_KEY" \
  -F "file=@data/your-iso-15189.pdf"
```

### Running locally without Docker

**Backend**
```bash
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp ../.env.example ../.env  # edit as needed; DATABASE_URL must point at a running Postgres instance
alembic upgrade head
uvicorn app.main:app --reload
```

**Frontend**
```bash
cd frontend
npm install
npm run dev
```

---

## Tech Stack

- **LangChain (LCEL)** – hybrid retrieval + generation pipeline
- **FastAPI** – backend API, Server-Sent Events streaming
- **Chroma + rank_bm25** – dense + sparse retrieval
- **sentence-transformers** – embeddings and cross-encoder reranking
- **PostgreSQL + SQLAlchemy + Alembic** – chat history persistence
- **Next.js + TypeScript + Tailwind CSS** – frontend
- **Docker Compose** – local orchestration
