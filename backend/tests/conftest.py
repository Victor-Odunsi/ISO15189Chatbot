import os

# app.core.config raises ValueError at import time if these are missing --
# set harmless placeholders before any test imports the app, since tests
# never make real LLM calls (they mock at the app.api.chat boundary).
os.environ.setdefault("GROQ_API_KEY", "test-key")
os.environ.setdefault("MISTRALAI_API_KEY", "test-key")
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://test:test@localhost:5432/test")
os.environ.setdefault("S3_BUCKET", "test-bucket")
# In-memory rate-limit storage for tests -- production uses Redis (see
# app.core.config.redis_url) so limits are shared across concurrent
# processes, but tests shouldn't need a real Redis instance to pass.
os.environ.setdefault("REDIS_URL", "memory://")
