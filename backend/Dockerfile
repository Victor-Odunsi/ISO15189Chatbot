FROM python:3.11.9-slim

WORKDIR /app/backend

RUN pip install --upgrade pip

COPY backend/requirements.txt .
RUN pip install --default-timeout=100 --no-cache-dir -r requirements.txt

# Pre-download the embedding + reranker models at build time. Without
# this, the first ingestion/query on a fresh container fetches them
# lazily from huggingface.co -- if that's slow or unreachable, app
# startup (via the lifespan's get_chroma() call) or the first request
# can hang for a long time rather than failing fast.
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; SentenceTransformer('BAAI/bge-small-en'); CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

COPY backend/ .

EXPOSE 8000

CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8000"]