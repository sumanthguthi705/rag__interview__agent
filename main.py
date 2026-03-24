"""
main.py — Interactive CLI for the RAG Interview Agent

Features:
  - Rich-formatted terminal UI with colors, panels, and markdown rendering
  - Session memory across the full conversation
  - Special commands: quiz, hint, topics, clear, exit
  - Active quiz indicator in the prompt
"""

import sys
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.text import Text
from rich.theme import Theme

# ── Config & Agent ────────────────────────────────────────────────────────────
from config import CHROMA_PERSIST_DIR, STUDY_TOPICS, validate_env
from agent import InterviewAgent, AgentState

# ── Custom Theme ──────────────────────────────────────────────────────────────
custom_theme = Theme({
    "user": "bold cyan",
    "agent": "bold white",
    "system": "bold yellow",
    "quiz": "bold magenta",
    "success": "bold green",
    "error": "bold red",
    "dim_info": "dim white",
})

console = Console(theme=custom_theme, highlight=False)

WELCOME_BANNER = """
# 🎓 Deep Learning Interview Preparation Agent

Welcome! I'm your AI-powered interview coach, trained on comprehensive deep learning study material.

## What I can do:
- **Answer questions** about any deep learning topic in our knowledge base
- **Generate quiz questions** to test your understanding
- **Evaluate your answers** with detailed feedback and a model response
- **Give hints** when you're stuck on a quiz question

## Quick commands:
| Command | Action |
|---------|--------|
| `quiz` | Get a new interview question |
| `hint` | Get a hint for the current question |
| `topics` | See all study topics |
| `clear` | Clear conversation history |
| `exit` | Quit the session |

---
*Type your question or command to begin. Good luck!* 🚀
"""

SPECIAL_COMMANDS = {"quiz", "hint", "topics", "clear", "exit", "quit", "help"}


def print_welcome() -> None:
    console.print()
    console.print(Panel(Markdown(WELCOME_BANNER), border_style="cyan", padding=(1, 2)))
    console.print()


def print_topics() -> None:
    console.print(Panel(
        "\n".join(f"  {topic}" for topic in STUDY_TOPICS),
        title="[bold cyan]Study Topics[/bold cyan]",
        border_style="cyan",
    ))
    console.print()


def format_agent_response(text: str) -> None:
    """Render agent response as rich Markdown."""
    console.print()
    console.print(Panel(
        Markdown(text),
        title="[bold white]🤖 Interview Agent[/bold white]",
        border_style="white",
        padding=(1, 2),
    ))
    console.print()


def build_prompt_text(state: AgentState) -> str:
    """Build the input prompt showing active quiz status."""
    if state.get("current_quiz_question"):
        topic = state.get("current_quiz_topic", "?")
        return f"[quiz]❓ Quiz Active [{topic}] > [/quiz]"
    return "[user]You > [/user]"


def validate_setup() -> None:
    """Check that ChromaDB has been populated before launching."""
    persist_path = Path(CHROMA_PERSIST_DIR)
    if not persist_path.exists() or not any(persist_path.iterdir()):
        console.print(Panel(
            "[error]ChromaDB vector store not found.[/error]\n\n"
            "Please run ingestion first:\n\n"
            "    [bold cyan]python ingest.py[/bold cyan]",
            border_style="red",
            title="[error]Setup Required[/error]",
        ))
        sys.exit(1)


def handle_special_command(
    command: str,
    agent: InterviewAgent,
    state: AgentState,
) -> tuple[bool, AgentState]:
    """
    Handle CLI special commands. Returns (handled: bool, updated_state).
    If handled=True, main loop should skip LLM processing.
    """
    cmd = command.strip().lower()

    if cmd in ("exit", "quit"):
        console.print()
        console.print(Rule("[dim_info]Session ended — Good luck with your interviews![/dim_info]"))
        sys.exit(0)

    elif cmd == "topics":
        print_topics()
        return True, state

    elif cmd == "clear":
        new_state = agent.get_initial_state()
        console.print("[system]✓ Conversation history cleared.[/system]\n")
        return True, new_state

    elif cmd == "help":
        print_welcome()
        return True, state

    elif cmd == "quiz":
        # Re-route as quiz_request
        return False, state  # Let LLM handle "quiz" as a natural language trigger

    elif cmd == "hint":
        if not state.get("current_quiz_question"):
            console.print("[system]⚠ No active quiz question. Ask for a quiz first![/system]\n")
            return True, state
        return False, state  # Let LLM handle hint routing

    return False, state


def run_session() -> None:
    """Main interactive session loop."""
    # Setup checks
    validate_env()
    validate_setup()

    # Init UI
    print_welcome()

    # Load agent (does vectorstore warmup)
    console.print("[dim_info]Initializing agent and loading embedding model...[/dim_info]")
    try:
        agent = InterviewAgent()
        console.print("[success]✓ Agent ready.[/success]\n")
    except RuntimeError as e:
        console.print(f"[error]Initialization failed:[/error] {e}")
        sys.exit(1)
    except Exception as e:
        console.print(f"[error]Unexpected error:[/error] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    state: AgentState = agent.get_initial_state()

    # Conversation loop
    while True:
        try:
            # Build contextual prompt
            prompt_text = build_prompt_text(state)
            user_input = Prompt.ask(Text.from_markup(prompt_text)).strip()

            if not user_input:
                continue

            # Check for special commands
            handled, state = handle_special_command(user_input, agent, state)
            if handled:
                continue

            # Show "thinking" indicator
            console.print("[dim_info]Thinking...[/dim_info]")

            # Call the LangGraph agent
            response, state = agent.chat(user_input, state)

            # Display response
            format_agent_response(response)

        except KeyboardInterrupt:
            console.print("\n\n[dim_info]Use 'exit' to quit gracefully.[/dim_info]")

        except Exception as exc:
            console.print(f"\n[error]Agent error:[/error] {exc}")
            console.print("[dim_info]The conversation state has been preserved. Try again.[/dim_info]\n")
            import traceback
            traceback.print_exc()


def main() -> None:
    run_session()


if __name__ == "__main__":
    main()
