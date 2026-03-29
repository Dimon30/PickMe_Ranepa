import os
from dotenv import load_dotenv

ENV_FILE = os.getenv("ENV_FILE", ".env")
load_dotenv(ENV_FILE)

BOT_TOKEN: str = os.getenv("BOT_TOKEN", "")
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql+psycopg2://user:password@localhost:5432/pickme")
MILVUS_URI: str = os.getenv("MILVUS_URI", "http://milvus:19530")
FASTAPI_URL: str = os.getenv("FASTAPI_URL", "http://fastapi:8000")
REDIS_URL: str = os.getenv("REDIS_URL", "redis://redis:6379/0")

GROQ_MODEL: str = "llama-3.3-70b-versatile"
EMBED_MODEL: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
EMBED_DIM: int = 768

FAQ_TOP_K: int = 3

HISTORY_MAX_MESSAGES: int = 10
HISTORY_TTL: int = 86400
