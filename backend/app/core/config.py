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
database_url = _getenv(
    'DATABASE_URL',
    'postgresql+asyncpg://iso15189:iso15189@localhost:5432/iso15189'
)
# Dense (pgvector) retrieval uses a separate sync engine/session; derive it
# from DATABASE_URL rather than requiring a second env var to keep in sync.
sync_database_url = database_url.replace('+asyncpg', '+psycopg')

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
