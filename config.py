"""
config.py — Centralized configuration for the RAG Interview Agent.

All tunable parameters live here. Override via environment variables (see .env.example).
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Project Paths ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
STUDY_MATERIAL_DIR = BASE_DIR / "study_material"
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", str(BASE_DIR / "chroma_db"))

# ── ChromaDB Settings ─────────────────────────────────────────────────────────
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "deep_learning_study")

# ── Embedding Model (local, no API key needed) ────────────────────────────────
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# ── LLM Settings ──────────────────────────────────────────────────────────────
LLM_MODEL = os.getenv("LLM_MODEL", "claude-3-5-haiku-20241022")
LLM_TEMPERATURE = 0.4          # Slight creativity for quiz variety
LLM_MAX_TOKENS = 1024

# ── Text Splitting ────────────────────────────────────────────────────────────
CHUNK_SIZE = 512               # Characters per chunk
CHUNK_OVERLAP = 64             # Overlap to preserve context at boundaries

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "5"))

# ── Study Topics (for `topics` command) ───────────────────────────────────────
STUDY_TOPICS = [
    "01 — Neural Networks Fundamentals",
    "02 — Backpropagation & Gradients",
    "03 — Convolutional Neural Networks (CNNs)",
    "04 — Recurrent Neural Networks & LSTMs",
    "05 — Transformers & Self-Attention",
    "06 — Optimization Algorithms",
    "07 — Regularization Techniques",
]

# ── Validation ────────────────────────────────────────────────────────────────
def validate_env() -> None:
    """Raise early with a helpful message if required env vars are missing."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise EnvironmentError(
            "\n[CONFIG ERROR] ANTHROPIC_API_KEY is not set.\n"
            "  1. Copy .env.example → .env\n"
            "  2. Add your key: ANTHROPIC_API_KEY=sk-ant-...\n"
            "  3. Re-run the script.\n"
        )
