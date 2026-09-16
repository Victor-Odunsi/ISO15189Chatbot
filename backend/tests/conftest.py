import os

# app.core.config raises ValueError at import time if these are missing --
# set harmless placeholders before any test imports the app, since tests
# never make real LLM calls (they mock at the app.api.chat boundary).
os.environ.setdefault("GROQ_API_KEY", "test-key")
os.environ.setdefault("MISTRALAI_API_KEY", "test-key")
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://test:test@localhost:5432/test")
