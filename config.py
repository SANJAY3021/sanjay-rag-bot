import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
TOP_K = int(os.getenv("TOP_K", "4"))

if CHUNK_SIZE <= 0:
    raise ValueError("CHUNK_SIZE must be greater than zero.")

if not 0 <= CHUNK_OVERLAP < CHUNK_SIZE:
    raise ValueError(
        "CHUNK_OVERLAP must be non-negative and smaller than CHUNK_SIZE."
    )

if TOP_K <= 0:
    raise ValueError("TOP_K must be greater than zero.")