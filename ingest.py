"""
ingest.py — RAG Ingestion Pipeline

Loads all Markdown study materials, splits them into overlapping chunks,
embeds with a local sentence-transformer model, and persists to ChromaDB.

Run this ONCE before using the agent:
    python ingest.py
"""

import sys
import time
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

# ── LangChain Imports ─────────────────────────────────────────────────────────
from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# ── Config ────────────────────────────────────────────────────────────────────
from config import (
    STUDY_MATERIAL_DIR,
    CHROMA_PERSIST_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

console = Console()


def load_documents() -> list:
    """Load all Markdown files from the study_material directory."""
    console.print("\n[bold cyan]Step 1/4 — Loading study material documents...[/bold cyan]")

    loader = DirectoryLoader(
        str(STUDY_MATERIAL_DIR),
        glob="**/*.md",
        loader_cls=UnstructuredMarkdownLoader,
        show_progress=True,
        use_multithreading=True,
    )

    docs = loader.load()

    # Enrich metadata with filename and topic
    for doc in docs:
        source_path = Path(doc.metadata.get("source", ""))
        doc.metadata["filename"] = source_path.name
        doc.metadata["topic"] = source_path.stem.replace("_", " ").title()

    console.print(f"  [green]✓[/green] Loaded [bold]{len(docs)}[/bold] documents")
    return docs


def split_documents(docs: list) -> list:
    """Split documents into overlapping chunks for retrieval."""
    console.print("\n[bold cyan]Step 2/4 — Chunking documents...[/bold cyan]")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""],
        length_function=len,
        add_start_index=True,
    )

    chunks = splitter.split_documents(docs)

    console.print(
        f"  [green]✓[/green] Created [bold]{len(chunks)}[/bold] chunks "
        f"(size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})"
    )

    # Print breakdown per document
    table = Table(title="Chunk Distribution", style="cyan")
    table.add_column("Document", style="white")
    table.add_column("Chunks", justify="right", style="yellow")

    from collections import Counter
    counts = Counter(c.metadata.get("filename", "unknown") for c in chunks)
    for filename, count in sorted(counts.items()):
        table.add_row(filename, str(count))

    console.print(table)
    return chunks


def build_embeddings() -> HuggingFaceEmbeddings:
    """Initialize the local embedding model."""
    console.print(f"\n[bold cyan]Step 3/4 — Loading embedding model: {EMBEDDING_MODEL}...[/bold cyan]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Downloading / loading model weights...", total=None)
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        progress.update(task, completed=True)

    # Quick sanity check
    test_embed = embeddings.embed_query("test")
    console.print(
        f"  [green]✓[/green] Model loaded — embedding dimension: [bold]{len(test_embed)}[/bold]"
    )
    return embeddings


def build_vectorstore(chunks: list, embeddings: HuggingFaceEmbeddings) -> Chroma:
    """Embed chunks and persist to ChromaDB."""
    console.print(f"\n[bold cyan]Step 4/4 — Building ChromaDB vector store...[/bold cyan]")
    console.print(f"  Collection : [bold]{COLLECTION_NAME}[/bold]")
    console.print(f"  Persist dir: [bold]{CHROMA_PERSIST_DIR}[/bold]")

    start = time.time()

    # Delete existing collection to avoid duplicates on re-ingestion
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
    )

    # Clear old data
    try:
        vectorstore.delete_collection()
        console.print("  [yellow]⚠[/yellow]  Cleared previous collection")
    except Exception:
        pass

    # Rebuild fresh
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_PERSIST_DIR,
    )

    elapsed = time.time() - start
    console.print(
        f"  [green]✓[/green] Indexed [bold]{len(chunks)}[/bold] chunks "
        f"in [bold]{elapsed:.1f}s[/bold]"
    )
    return vectorstore


def verify_vectorstore(vectorstore: Chroma) -> None:
    """Run a quick retrieval sanity check."""
    console.print("\n[bold cyan]Verification — Running retrieval sanity check...[/bold cyan]")

    test_queries = [
        "What is backpropagation?",
        "How does the LSTM forget gate work?",
        "What is the softmax function?",
    ]

    for query in test_queries:
        results = vectorstore.similarity_search(query, k=1)
        if results:
            snippet = results[0].page_content[:80].replace("\n", " ")
            topic = results[0].metadata.get("topic", "unknown")
            console.print(
                f"  [green]✓[/green] [dim]{query}[/dim]\n"
                f"    → [{topic}] {snippet}..."
            )
        else:
            console.print(f"  [red]✗[/red] No results for: {query}")


def main() -> None:
    console.print(
        Panel.fit(
            "[bold white]RAG Interview Agent — Ingestion Pipeline[/bold white]\n"
            "[dim]Loads, chunks, embeds, and indexes study material into ChromaDB[/dim]",
            border_style="cyan",
        )
    )

    # Validate study material exists
    if not STUDY_MATERIAL_DIR.exists():
        console.print(
            f"[bold red]ERROR:[/bold red] Study material directory not found: {STUDY_MATERIAL_DIR}"
        )
        sys.exit(1)

    md_files = list(STUDY_MATERIAL_DIR.glob("**/*.md"))
    if not md_files:
        console.print(
            f"[bold red]ERROR:[/bold red] No .md files found in {STUDY_MATERIAL_DIR}"
        )
        sys.exit(1)

    console.print(f"  Found [bold]{len(md_files)}[/bold] Markdown files to ingest")

    try:
        docs = load_documents()
        chunks = split_documents(docs)
        embeddings = build_embeddings()
        vectorstore = build_vectorstore(chunks, embeddings)
        verify_vectorstore(vectorstore)

        console.print(
            Panel.fit(
                "[bold green]✓ Ingestion complete![/bold green]\n\n"
                f"Indexed [bold]{len(chunks)}[/bold] chunks from "
                f"[bold]{len(docs)}[/bold] documents into ChromaDB.\n"
                "Run [bold cyan]python main.py[/bold cyan] to start the interview agent.",
                border_style="green",
            )
        )

    except Exception as exc:
        console.print(f"\n[bold red]INGESTION FAILED:[/bold red] {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
