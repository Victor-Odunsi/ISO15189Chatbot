FROM python:3.11.9-slim

# Lambda Web Adapter: proxies Lambda invokes (including RESPONSE_STREAM
# mode, configured on the function's Function URL + AWS_LWA_INVOKE_MODE)
# to the normal uvicorn process below -- no ASGI/Lambda handler code
# needed, api/chat.py's StreamingResponse works unmodified.
COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:1.0.1 /lambda-adapter /opt/extensions/lambda-adapter

WORKDIR /app/backend

RUN pip install --upgrade pip

# requirements.txt pins torch==2.8.0+cpu (the default PyPI wheel is the CUDA
# build, several GB of nvidia-* libraries a CPU-only Lambda never uses).
# Installing it in its own earlier layer is just a build-cache optimisation:
# torch is by far the largest dependency and changes least often, so editing
# any other requirement doesn't re-download it.
RUN pip install --default-timeout=100 --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu torch==2.8.0+cpu

COPY backend/requirements.txt .
RUN pip install --default-timeout=100 --no-cache-dir -r requirements.txt

# Bake both models into the image so no request ever downloads them from
# HF Hub at runtime (a suspect in retrieval hanging until Lambda's 120s
# timeout, and needless per-container cold-start work either way).
# HF_HUB_OFFLINE=1 makes any accidental runtime fetch fail fast instead of
# hanging. Read-only at runtime is fine: loading from a cache doesn't write.
ENV HF_CACHE_DIR=/opt/hf_cache
RUN python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('BAAI/bge-small-en', cache_folder='/opt/hf_cache'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', cache_folder='/opt/hf_cache')"
ENV HF_HUB_OFFLINE=1

COPY backend/ .

EXPOSE 8000

# Migrations are a deploy-time step (run `alembic upgrade head` against
# the RDS endpoint after `sam deploy`), not run here -- doing it on every
# cold start adds latency and risks concurrent-cold-start migration races.
# Same entrypoint locally and on Lambda -- LWA (added above) is configured
# via AWS_LWA_PORT=8000 (see template.yaml) to proxy to this port rather
# than its own default of 8080, so nothing here needs to change per target.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
