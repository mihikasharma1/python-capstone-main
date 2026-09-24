import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY not set. Copy .env.example to .env and add your key."
    )

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "data" / "docs"
CHROMA_DIR = BASE_DIR / "chroma_store"

GENERATION_MODEL = "gemini-3.6-flash"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

TOP_K = 3
RELEVANCE_THRESHOLD = 0.45

DB_PATH = BASE_DIR / "data" / "company.db"