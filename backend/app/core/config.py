import os
from dotenv import load_dotenv

load_dotenv()

groq_api_key = os.getenv('GROQ_API_KEY')
mistral_api_key = os.getenv('MISTRALAI_API_KEY')
database_url = os.getenv(
    'DATABASE_URL',
    'postgresql+asyncpg://iso15189:iso15189@localhost:5432/iso15189'
)
# Dense (pgvector) retrieval uses a separate sync engine/session; derive it
# from DATABASE_URL rather than requiring a second env var to keep in sync.
sync_database_url = database_url.replace('+asyncpg', '+psycopg')

s3_bucket = os.getenv('S3_BUCKET')
aws_region = os.getenv('AWS_REGION', 'us-east-1')

if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in environment variables")

if not mistral_api_key:
    raise ValueError("MISTRALAI_API_KEY not found in environment variables")
