"""
config.py — Centralized configuration for the RAG Interview Agent.

All tunable parameters live here. Override via environment variables (see .env.example).
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent
STUDY_MATERIAL_DIR = BASE_DIR / "study_material"
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", str(BASE_DIR / "chroma_db"))

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "deep_learning_study")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

#LLM_MODEL = os.getenv("LLM_MODEL", "groq/compound-mini")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
LLM_TEMPERATURE = 0.4
LLM_MAX_TOKENS = 1024

CHUNK_SIZE = 512
CHUNK_OVERLAP = 64

TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "5"))

STUDY_TOPICS = [
    "01 — Neural Networks Fundamentals",
    "02 — Backpropagation & Gradients",
    "03 — Convolutional Neural Networks (CNNs)",
    "04 — Recurrent Neural Networks & LSTMs",
    "05 — Transformers & Self-Attention",
    "06 — Optimization Algorithms",
    "07 — Regularization Techniques",
]

def validate_env() -> None:
    if not os.getenv("GROQ_API_KEY"):
        raise EnvironmentError(
            "\n[CONFIG ERROR] GROQ_API_KEY is not set.\n"
            "  1. Copy .env.example → .env\n"
            "  2. Add your key: GROQ_API_KEY=gsk_...\n"
            "  3. Re-run the script.\n"
        )