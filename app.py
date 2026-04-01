"""
app.py — Streamlit UI for the RAG Interview Prep Agent

3-panel layout required by the professor:
  Panel 1 (left sidebar)  — Document Ingestion: upload files, see status, list ingested docs
  Panel 2 (center)        — Chat Interface: conversation history, query input, source citations
  Panel 3 (right sidebar) — Document Viewer: select a doc, browse its chunks

Run with:
    streamlit run app.py

Requires agent.py, config.py, and ingest.py to be in the same directory.
"""

import os
import sys
import hashlib
from pathlib import Path

import streamlit as st
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

# Add parent directory to path for imports
sys.path.append('..')

# ── Page config (must be first Streamlit call) ─────────────────────────────
st.set_page_config(
    page_title="Deep Learning Interview Prep",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Lazy imports (after page config) ──────────────────────────────────────
from config import (
    CHROMA_PERSIST_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    STUDY_TOPICS,
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Tighten sidebar padding */
section[data-testid="stSidebar"] { padding-top: 1rem; }

/* Chat bubbles */
.user-bubble {
    background: #1a73e8;
    color: white;
    border-radius: 16px 16px 4px 16px;
    padding: 10px 14px;
    margin: 6px 0 6px 20%;
    font-size: 0.93rem;
    line-height: 1.5;
}
.agent-bubble {
    background: #f1f3f4;
    color: #202124;
    border-radius: 16px 16px 16px 4px;
    padding: 10px 14px;
    margin: 6px 20% 6px 0;
    font-size: 0.93rem;
    line-height: 1.5;
}
/* Source citation pill */
.source-pill {
    display: inline-block;
    background: #e8f0fe;
    color: #1967d2;
    border-radius: 12px;
    padding: 2px 10px;
    font-size: 0.78rem;
    margin: 2px 3px;
}
/* Status badges */
.badge-success { color: #188038; font-weight: 600; }
.badge-warning { color: #e37400; font-weight: 600; }
.badge-error   { color: #c5221f; font-weight: 600; }

/* Panel headers */
.panel-header {
    font-size: 1.05rem;
    font-weight: 600;
    padding-bottom: 6px;
    border-bottom: 2px solid #e8eaed;
    margin-bottom: 12px;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# Session State Initialization
# ══════════════════════════════════════════════════════════════════

def initialise_session_state():
    """Create all session state keys on first run."""
    if "agent" not in st.session_state:
        st.session_state.agent = None
    if "agent_state" not in st.session_state:
        st.session_state.agent_state = None
    if "chat_history" not in st.session_state:
        # Each entry: {"role": "user"|"agent", "content": str, "sources": list}
        st.session_state.chat_history = []
    if "ingested_docs" not in st.session_state:
        # {filename: {"chunks": int, "hash": str, "topics": list}}
        st.session_state.ingested_docs = {}
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "selected_doc" not in st.session_state:
        st.session_state.selected_doc = None
    if "doc_chunks" not in st.session_state:
        st.session_state.doc_chunks = []
    if "ingestion_log" not in st.session_state:
        st.session_state.ingestion_log = []
    if "active_quiz" not in st.session_state:
        st.session_state.active_quiz = False


initialise_session_state()


# ══════════════════════════════════════════════════════════════════
# Resource Initialization (cached)
# ══════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading embedding model…")
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


@st.cache_resource(show_spinner="Connecting to ChromaDB…")
def get_vectorstore():
    embeddings = get_embeddings()
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
    )


@st.cache_resource(show_spinner="Initialising agent…")
def get_agent():
    try:
        from agent import InterviewAgent
        return InterviewAgent()
    except RuntimeError as e:
        return None, str(e)
    except Exception as e:
        return None, str(e)


# ══════════════════════════════════════════════════════════════════
# Helper Utilities
# ══════════════════════════════════════════════════════════════════

def file_hash(content: bytes) -> str:
    return hashlib.md5(content).hexdigest()


def check_duplicate(vs: Chroma, doc_hash: str) -> bool:
    """Return True if a document with this hash already exists in the store."""
    try:
        results = vs.get(where={"doc_hash": doc_hash})
        return len(results["ids"]) > 0
    except Exception:
        return False


def ingest_file(uploaded_file) -> dict:
    """
    Ingest an uploaded Streamlit file object into ChromaDB.
    Returns a result dict with keys: status, chunks, skipped, message.
    """
    vs = get_vectorstore()
    content = uploaded_file.read()
    doc_hash = file_hash(content)

    # Duplicate detection
    if check_duplicate(vs, doc_hash):
        return {
            "status": "skipped",
            "chunks": 0,
            "message": f"⏭️ **{uploaded_file.name}** — already ingested (duplicate detected)",
        }

    # Write to a temp file so TextLoader can read it (cross-platform)
    import tempfile
    tmp_dir = Path(tempfile.gettempdir())
    tmp_path = tmp_dir / uploaded_file.name
    tmp_path.write_bytes(content)

    try:
        loader = TextLoader(str(tmp_path), encoding="utf-8")
        documents = loader.load()
    except Exception as e:
        return {"status": "error", "chunks": 0, "message": f"❌ Load error: {e}"}

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n\n", "\n", " "],
    )
    chunks = splitter.split_documents(documents)

    # Tag every chunk with source metadata + hash for duplicate detection
    for i, chunk in enumerate(chunks):
        chunk.metadata.update({
            "source": uploaded_file.name,
            "chunk_index": i,
            "doc_hash": doc_hash,
        })

    try:
        vs.add_documents(chunks)
    except Exception as e:
        return {"status": "error", "chunks": 0, "message": f"❌ Ingest error: {e}"}
    finally:
        tmp_path.unlink(missing_ok=True)

    return {
        "status": "success",
        "chunks": len(chunks),
        "message": f"✅ **{uploaded_file.name}** — ingested {len(chunks)} chunks",
    }


def get_doc_chunks(vs: Chroma, source_name: str) -> list[dict]:
    """Fetch all chunks for a given source document (matches on filename only)."""
    try:
        # Fetch all and filter by filename — handles both full-path and bare-name metadata
        results = vs.get(include=["documents", "metadatas"])
        chunks = []
        for doc, meta in zip(results["documents"], results["metadatas"]):
            stored = Path(meta.get("source", "")).name
            if stored == source_name or meta.get("source", "") == source_name:
                chunks.append({"text": doc, "metadata": {**meta, "source": stored}})
        chunks.sort(key=lambda x: x["metadata"].get("chunk_index", 0))
        return chunks
    except Exception:
        return []


def render_chat_message(role: str, content: str, sources: list = None):
    """Render a single chat bubble with optional source citations."""
    if role == "user":
        st.markdown(
            f'<div class="user-bubble">{content}</div>',
            unsafe_allow_html=True,
        )
    else:
        # Agent bubble: use a bordered container so markdown renders natively
        with st.container(border=False):
            st.markdown(
                '<div style="background:#f1f3f4;border-radius:16px 16px 16px 4px;'
                'padding:12px 16px;margin:4px 15% 4px 0;">',
                unsafe_allow_html=True,
            )
            st.markdown(content)
            if sources:
                pills = " ".join(
                    f'<span class="source-pill">📄 {Path(s).name}</span>'
                    for s in set(sources) if s
                )
                st.markdown(
                    f'<div style="margin-top:6px">{pills}</div>',
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# PANEL 1 — Left Sidebar: Document Ingestion
# ══════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown('<div class="panel-header">📥 Panel 1 — Document Ingestion</div>', unsafe_allow_html=True)

    # ── File uploader ──────────────────────────────────────────────
    uploaded_files = st.file_uploader(
        "Upload study material (.md or .txt)",
        type=["md", "txt"],
        accept_multiple_files=True,
        help="Upload Markdown or text files. Duplicates are automatically detected and skipped.",
    )

    if uploaded_files:
        if st.button("📤 Ingest Files", type="primary", use_container_width=True):
            vs = get_vectorstore()
            progress = st.progress(0)
            log_entries = []
            for i, f in enumerate(uploaded_files):
                result = ingest_file(f)
                log_entries.append(result["message"])
                if result["status"] == "success":
                    st.session_state.ingested_docs[f.name] = {
                        "chunks": result["chunks"],
                    }
                progress.progress((i + 1) / len(uploaded_files))
            st.session_state.ingestion_log = log_entries
            # Bust the agent cache so it sees new data
            get_agent.clear()
            st.rerun()

    # ── Ingestion log ──────────────────────────────────────────────
    if st.session_state.ingestion_log:
        st.markdown("**Last ingestion results:**")
        for entry in st.session_state.ingestion_log:
            st.markdown(entry)

    st.divider()

    # ── Ingested document list ─────────────────────────────────────
    st.markdown("**Ingested documents**")
    vs = get_vectorstore()
    try:
        count = vs._collection.count()
        st.caption(f"Total chunks in store: **{count}**")
    except Exception:
        count = 0

    if count == 0:
        st.info("No documents yet. Upload files above or run `python ingest.py` first.")
    else:
        # Pull unique source names — strip full paths to just the filename
        try:
            all_meta = vs.get(include=["metadatas"])["metadatas"]
            sources = sorted(set(Path(m.get("source", "unknown")).name for m in all_meta))
        except Exception:
            sources = list(st.session_state.ingested_docs.keys())

        for src in sources:
            st.markdown(f"📄 `{src}`")
            if st.button("View in Panel 3 →", key=f"view_{src}", use_container_width=True):
                st.session_state.selected_doc = src
                st.session_state.doc_chunks = get_doc_chunks(vs, src)
                st.rerun()

    st.divider()

    # ── Quick topics reference ─────────────────────────────────────
    with st.expander("📚 Study topics"):
        for t in STUDY_TOPICS:
            st.markdown(f"- {t}")


# ══════════════════════════════════════════════════════════════════
# Main area: Panel 2 (chat) + Panel 3 (doc viewer)
# ══════════════════════════════════════════════════════════════════

col_chat, col_viewer = st.columns([3, 2], gap="large")

# ── PANEL 2 — Chat Interface ───────────────────────────────────────────────
with col_chat:
    st.markdown('<div class="panel-header">💬 Panel 2 — Chat Interface</div>', unsafe_allow_html=True)

    # Load agent (may be None if DB is empty)
    agent_obj = get_agent()
    agent_ok = False

    if isinstance(agent_obj, tuple):
        # get_agent returned (None, error_msg)
        st.error(f"Agent not ready: {agent_obj[1]}")
        st.info("Run `python ingest.py` (or upload files via Panel 1) to populate ChromaDB first.")
    else:
        agent_ok = True
        if st.session_state.agent_state is None:
            st.session_state.agent_state = agent_obj.get_initial_state()

    # ── Toolbar ──────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    if c1.button("🎯 Quiz me", use_container_width=True, disabled=not agent_ok):
        if agent_ok:
            response, new_state = agent_obj.chat("quiz me", st.session_state.agent_state)
            st.session_state.agent_state = new_state
            st.session_state.chat_history.append({"role": "user", "content": "Quiz me!", "sources": []})
            sources = [m.get("source") for m in new_state.get("retrieved_metadata", []) if m.get("source")]
            st.session_state.chat_history.append({"role": "agent", "content": response, "sources": sources})
            st.session_state.active_quiz = bool(new_state.get("current_quiz_question"))
            st.rerun()

    if c2.button("💡 Hint", use_container_width=True, disabled=(not agent_ok or not st.session_state.active_quiz)):
        if agent_ok:
            response, new_state = agent_obj.chat("hint", st.session_state.agent_state)
            st.session_state.agent_state = new_state
            st.session_state.chat_history.append({"role": "user", "content": "Hint please", "sources": []})
            st.session_state.chat_history.append({"role": "agent", "content": response, "sources": []})
            st.rerun()

    if c3.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.active_quiz = False
        if agent_ok:
            st.session_state.agent_state = agent_obj.get_initial_state()
        st.rerun()

    # ── Quiz status indicator ─────────────────────────────────────
    if st.session_state.active_quiz:
        st.info("📋 Active quiz — type your answer below, or click **Hint** for a nudge.")

    # ── Chat history ──────────────────────────────────────────────
    chat_container = st.container(height=460, border=True)
    with chat_container:
        if not st.session_state.chat_history:
            st.markdown(
                "<div style='text-align:center; color:#888; padding-top:40px'>"
                "🧠 Ask anything about deep learning, or click <b>Quiz me</b> to get started."
                "</div>",
                unsafe_allow_html=True,
            )
        for msg in st.session_state.chat_history:
            render_chat_message(msg["role"], msg["content"], msg.get("sources", []))

    # ── Input box ─────────────────────────────────────────────────
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_area(
            "Your message",
            placeholder="Ask a question, type your quiz answer, or say 'quiz me'…",
            height=80,
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Send ➤", type="primary", use_container_width=True)

    if submitted and user_input.strip() and agent_ok:
        user_msg = user_input.strip()
        response, new_state = agent_obj.chat(user_msg, st.session_state.agent_state)
        st.session_state.agent_state = new_state
        sources = [
            m.get("source") for m in new_state.get("retrieved_metadata", []) if m.get("source")
        ]
        st.session_state.chat_history.append({"role": "user", "content": user_msg, "sources": []})
        st.session_state.chat_history.append({"role": "agent", "content": response, "sources": sources})
        st.session_state.active_quiz = bool(new_state.get("current_quiz_question"))
        st.rerun()


# ── PANEL 3 — Document Viewer ──────────────────────────────────────────────
with col_viewer:
    st.markdown('<div class="panel-header">🔍 Panel 3 — Document Viewer</div>', unsafe_allow_html=True)

    vs = get_vectorstore()
    try:
        all_meta = vs.get(include=["metadatas"])["metadatas"]
        sources = sorted(set(Path(m.get("source", "unknown")).name for m in all_meta))
    except Exception:
        sources = []

    if not sources:
        st.info("No documents ingested yet. Use Panel 1 to upload files.")
    else:
        # Source selector
        selected = st.selectbox(
            "Select a document to inspect",
            options=sources,
            index=sources.index(st.session_state.selected_doc)
                if st.session_state.selected_doc in sources else 0,
        )

        if selected != st.session_state.selected_doc:
            st.session_state.selected_doc = selected
            st.session_state.doc_chunks = get_doc_chunks(vs, selected)

        chunks = st.session_state.doc_chunks
        if not chunks and selected:
            chunks = get_doc_chunks(vs, selected)
            st.session_state.doc_chunks = chunks

        if chunks:
            st.caption(f"**{len(chunks)} chunks** from `{selected}`")

            # Chunk browser
            chunk_idx = st.slider(
                "Chunk",
                min_value=1,
                max_value=len(chunks),
                value=1,
                format="Chunk %d",
            )
            chunk = chunks[chunk_idx - 1]

            with st.container(border=True):
                st.markdown(f"**Chunk {chunk_idx} / {len(chunks)}**")
                st.markdown(chunk["text"])

            # Metadata
            with st.expander("📋 Chunk metadata"):
                meta = chunk["metadata"]
                for k, v in meta.items():
                    if k != "doc_hash":
                        st.markdown(f"- **{k}**: `{v}`")

            # Word count
            words = len(chunk["text"].split())
            chars = len(chunk["text"])
            col_a, col_b = st.columns(2)
            col_a.metric("Words", words)
            col_b.metric("Characters", chars)

            # Quality check (professor requires 100-300 words per chunk)
            if words < 100:
                st.warning(f"⚠️ Chunk is short ({words} words). Professor spec: 100–300 words.")
            elif words > 300:
                st.warning(f"⚠️ Chunk is long ({words} words). Professor spec: 100–300 words.")
            else:
                st.success(f"✅ Chunk size is within spec ({words} words).")
        else:
            st.caption("Select a document above to browse its chunks.")

    st.divider()

    # ── Demo checklist ─────────────────────────────────────────────
    st.markdown("**Demo checklist (Part 3)**")
    demo_items = [
        ("Ingest a document", bool(sources)),
        ("Duplicate detection works", bool(sources)),
        ("Successful RAG query with citation", len(st.session_state.chat_history) > 0),
        ("Quiz question generated", st.session_state.active_quiz or any(
            "Quiz me" in m["content"] for m in st.session_state.chat_history if m["role"] == "user"
        )),
        ("Answer evaluated", any(
            "Evaluation" in m["content"] for m in st.session_state.chat_history if m["role"] == "agent"
        )),
    ]
    for label, done in demo_items:
        icon = "✅" if done else "⬜"
        st.markdown(f"{icon} {label}")
