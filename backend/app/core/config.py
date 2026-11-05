import os
from dotenv import load_dotenv

load_dotenv()


def _getenv(key: str, default: str | None = None) -> str | None:
    # GitHub Actions secrets pasted via the web UI can silently pick up a
    # trailing newline, which breaks URL parsing (e.g. a DB name of
    # "postgres\n") in ways that are very confusing to trace back to this.
    value = os.getenv(key, default)
    return value.strip() if value is not None else value


groq_api_key = _getenv('GROQ_API_KEY')
mistral_api_key = _getenv('MISTRALAI_API_KEY')
# Optional: models from a provider whose key is unset are skipped (see
# services/llm.py), so the app still runs on Groq/Mistral without it.
gemini_api_key = _getenv('GEMINI_API_KEY') or None

# Ordered "provider:model" fallback chains -- the first entry is primary,
# later ones are tried on rate limits / timeouts / provider errors (with no
# backoff retries in between, so a 429 spills over in ~100ms).
#
# Defaults are latency-first with capacity spillover, from measured benchmarks
# (llm-evals/harness/bench_router.py, bench_generation.py, Nov 2026):
#   router TTFT p50/p95: gpt-oss-120b 474/873ms (19/20 intents), gpt-oss-20b
#     692/2529ms (19/20), gemini-3.5-flash-lite 1273/7715ms (14/20, 5 errors)
#   generation TTFT p50: gpt-oss-120b 750ms, gpt-oss-20b 848ms,
#     gemini-3.5-flash-lite 1533ms (max ~7s)
# Groq's free tier caps each model at 8K tokens/min (per-model buckets), so
# 120b -> 20b doubles Groq capacity; Gemini's free tier (250K TPM, 10 RPM)
# absorbs bursts beyond that, at higher latency.
router_models = _getenv(
    'ROUTER_MODELS',
    'groq:openai/gpt-oss-120b,groq:openai/gpt-oss-20b,gemini:gemini-3.5-flash-lite',
)
generation_models = _getenv(
    'GENERATION_MODELS',
    'groq:openai/gpt-oss-120b,groq:openai/gpt-oss-20b,gemini:gemini-3.5-flash-lite',
)

database_url = _getenv(
    'DATABASE_URL',
    'postgresql+asyncpg://iso15189:iso15189@localhost:5432/iso15189'
)
# Dense (pgvector) retrieval uses a separate sync engine/session; derive it
# from DATABASE_URL rather than requiring a second env var to keep in sync.
sync_database_url = database_url.replace('+asyncpg', '+psycopg')
# The async engine also runs on psycopg (not asyncpg) so it can keep a real
# connection pool against Supabase's transaction-mode pooler -- see
# app/db/session.py. DATABASE_URL keeps its +asyncpg form for compatibility.
async_database_url = sync_database_url

s3_bucket = _getenv('S3_BUCKET')
aws_region = _getenv('AWS_REGION', 'us-east-1')
sqs_queue_url = _getenv('SQS_QUEUE_URL')

# Rate-limit storage: Redis-backed so limits are shared across concurrent
# Lambda invocations (separate processes with no shared memory), not just
# per-container as an in-memory limiter would be.
redis_url = _getenv('REDIS_URL', 'redis://localhost:6379')

if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in environment variables")

if not mistral_api_key:
    raise ValueError("MISTRALAI_API_KEY not found in environment variables")
