# ISO 15189:2022 RAG Chatbot – Laboratory Quality Management Assistant

## Overview

As a **medical laboratory scientist**, I recognize how critical **Quality Management Systems (QMS)** are for ensuring accurate results, patient safety, and accreditation. However, navigating the **ISO 15189:2022** standard can be challenging for professionals seeking practical guidance.

This project uses **Retrieval-Augmented Generation (RAG)** to build a chatbot that provides **clause-specific, cited answers** from the ISO 15189:2022 document. Users can ask questions such as:
> *"What does ISO 15189 say about equipment calibration?"*

...or ask for the same content as an audit checklist or a Standard Operating Procedure draft. The chatbot retrieves the relevant clauses, generates a grounded response, and cites the source document and page for every answer.

---

## Features

- Hybrid retrieval: dense (pgvector) + sparse (BM25) search, merged with Reciprocal Rank Fusion and reranked with a local cross-encoder
- Query rewriting and intent classification (general question / checklist / SOP) via a single structured-output LLM call
- Real token-by-token streaming (Server-Sent Events), with source citations delivered before the answer streams in
- Persistent, multi-session chat history (Postgres), with a Next.js UI for browsing and resuming past conversations
- Admin endpoint for queuing new source documents for ingestion, protected by a shared secret

---

## Architecture

```
frontend/   Next.js (App Router, TypeScript, Tailwind) — chat UI, SSE client. Deployed to Vercel.
backend/    FastAPI, deployed to AWS Lambda (Function URL, streaming) via SAM
  app/api/       routes (chat, admin, sessions)
  app/services/  hybrid retrieval, ingestion, the LCEL chain, LLM client
  app/db/        Postgres (pgvector + chat history) via SQLAlchemy + Alembic
  app/core/      config, security
  app/worker/    SQS-triggered ingestion worker (separate Lambda function)
```

- **LLM**: Groq (`openai/gpt-oss-120b`), with Mistral (`mistral-small-latest`) as a fallback — both free-tier
- **Embeddings**: `BAAI/bge-small-en` (HuggingFace, local/CPU)
- **Reranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2` (local/CPU)
- **Vector store + chat history**: Postgres (pgvector extension)
- **Sparse retrieval corpus**: S3 (BM25 pickle)
- **Rate limiting**: Redis
- **Background ingestion**: S3 upload → SQS message → separate worker Lambda (see [Deploying to AWS Lambda](#deploying-to-aws-lambda))

---

## Getting Started

### Prerequisites

- Python 3.11+, Node.js 20+
- A [Groq](https://console.groq.com) API key (and optionally a [Mistral](https://mistral.ai) key as fallback) — both have usable free tiers
- Docker, if running via `docker-compose` (recommended)
- An AWS account with an S3 bucket and SQS queue (used even for local dev — see [Local dev note](#local-dev-note-on-s3sqs) below)
- Your own copy of the ISO 15189:2022 standard as a PDF — it's a licensed document and isn't included in this repo

### Running with Docker Compose

```bash
cp .env.example .env
# fill in GROQ_API_KEY / MISTRALAI_API_KEY / S3_BUCKET / AWS credentials in .env

cp your-iso-15189.pdf data/

docker compose up --build
# backend:  http://localhost:8000
# frontend: http://localhost:3000
```

This starts Postgres (with pgvector), Redis, the backend, and the frontend. Run migrations once against it:

```bash
docker compose exec backend alembic upgrade head
```

#### Local dev note on S3/SQS

The admin upload endpoint always uploads to S3 and queues an SQS message — there's no local-disk fallback, so local dev uses a real (free-tier) S3 bucket and SQS queue rather than an emulator. The queue's *consumer*, though, only exists once deployed (it's a separate Lambda function triggered by SQS — see below), so for local ingestion, skip the endpoint and ingest directly instead:

```bash
docker compose exec backend python -c "from app.services.ingestion import run_ingestion; run_ingestion('./data')"
```

### Running locally without Docker

**Backend**
```bash
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp ../.env.example ../.env  # edit as needed; DATABASE_URL must point at a Postgres instance with the pgvector extension installed, REDIS_URL at a running Redis
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

## Deploying to AWS Lambda

The backend deploys as two Lambda functions via [AWS SAM](https://docs.aws.amazon.com/serverless-application-model/) (`backend/template.yaml`): a streaming API function (FastAPI + [Lambda Web Adapter](https://github.com/awslabs/aws-lambda-web-adapter), behind a Function URL in `RESPONSE_STREAM` mode — real SSE streaming, not buffered) and a separate SQS-triggered ingestion worker (`backend/Dockerfile.worker`). Postgres (with pgvector) and Redis are **not** AWS-native here — RDS has no meaningful free tier for this workload and ElastiCache needs a VPC (extra cold-start latency from ENI attachment), so both are external, serverless-friendly providers instead:

- **Postgres**: [Supabase](https://supabase.com) (pgvector supported natively). Use the **connection pooler** string (Supavisor, transaction mode, port 6543), not the direct connection — Lambda can run many concurrent invocations, each wanting its own connection, which exhausts Postgres's connection limit fast against a direct connection. The app already disables server-side prepared statements on both engines (`app/db/session.py`) specifically because pooled connections require that.
- **Redis**: [Upstash](https://upstash.com) — reachable without a VPC, so no ENI cold-start penalty.
- **S3 bucket**: a plain AWS S3 bucket (the template takes its name as a parameter rather than creating it).

```bash
cd backend
sam build
sam deploy --guided
# provide: DatabaseUrl (Supabase pooler string, postgresql+asyncpg://...), S3Bucket,
# RedisUrl (Upstash), GroqApiKey, MistralApiKey, AdminApiKey, FrontendOrigin

# migrations are a deploy-time step, not run on cold start -- run once per deploy:
DATABASE_URL=<your Supabase pooler url> alembic upgrade head
```

`sam deploy` prints `ApiFunctionUrl` in its outputs — set that as `NEXT_PUBLIC_API_URL` in the Vercel project settings for the frontend.

**Known tradeoff**: models are *not* baked into the image (kept build light per this being a free/portfolio project), so each cold start pays the HuggingFace download cost for the embedding + reranker models — expect the first request after idle to be noticeably slower.

---

## Tech Stack

- **LangChain (LCEL)** – hybrid retrieval + generation pipeline
- **FastAPI** – backend API, Server-Sent Events streaming
- **pgvector + rank_bm25** – dense + sparse retrieval
- **sentence-transformers** – embeddings and cross-encoder reranking
- **PostgreSQL + SQLAlchemy + Alembic** – vector store and chat history persistence
- **Redis** – shared rate-limit storage
- **Next.js + TypeScript + Tailwind CSS** – frontend
- **AWS Lambda + Lambda Web Adapter + SAM, S3, SQS** – backend hosting
- **Vercel** – frontend hosting
- **Docker Compose** – local orchestration
