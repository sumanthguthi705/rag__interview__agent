"""
agent.py — LangGraph RAG Interview Agent

State machine with 5 nodes:
  1. classify_intent   — Determines if user wants to chat, get a quiz, or answer one
  2. retrieve          — Fetches relevant chunks from ChromaDB
  3. rag_respond       — Generates a grounded answer for conversational queries
  4. generate_quiz     — Creates a technical interview question from retrieved docs
  5. evaluate_answer   — Grades the user's answer against retrieved context

Graph flow:
  START → classify_intent → [retrieve] → {rag_respond | generate_quiz | evaluate_answer} → END
"""

from __future__ import annotations

import os
from typing import Annotated, Literal, Optional
from typing_extensions import TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from config import (
    CHROMA_PERSIST_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_TEMPERATURE,
    TOP_K_RETRIEVAL,
    validate_env,
)

# ── Type Definitions ──────────────────────────────────────────────────────────

Intent = Literal["chat", "quiz_request", "quiz_answer", "hint_request"]


class AgentState(TypedDict):
    """Full state passed between LangGraph nodes."""
    messages: Annotated[list[BaseMessage], add_messages]
    intent: Optional[Intent]
    retrieved_docs: list[str]                   # Raw chunk texts for context
    retrieved_metadata: list[dict]              # Source metadata per chunk
    current_quiz_question: Optional[str]        # Active quiz question
    current_quiz_context: Optional[str]         # Supporting docs for active quiz
    current_quiz_topic: Optional[str]           # Topic label for current quiz
    last_response: Optional[str]


# ── Prompts ───────────────────────────────────────────────────────────────────

CLASSIFIER_SYSTEM = """You are an intent classifier for an interview preparation assistant.
Classify the user's latest message into exactly one of these intents:

- quiz_request: User wants to be asked a new interview question (e.g., "quiz me", "give me a question", "test me", "ask me something about CNNs")
- quiz_answer: User is answering a previously asked quiz question (context: there IS an active quiz question)
- hint_request: User explicitly asks for a hint on the current quiz question
- chat: User wants information, explanation, or is having a general conversation

Rules:
- If there is NO active quiz question, NEVER classify as quiz_answer
- If user says "explain X" or "what is X", classify as chat
- If user says "quiz me on X" or "test me", classify as quiz_request
- Respond with ONLY the intent label (one word or underscore-joined). No explanation.
"""

RAG_RESPOND_SYSTEM = """You are an expert deep learning educator and technical interview coach.
Answer the user's question using ONLY the provided context from our study materials.

Guidelines:
- Be precise and technical; use correct terminology
- Reference specific concepts from the context
- If the context does not contain enough information, say so honestly
- Structure longer answers with clear sections
- Conclude with 1-2 key takeaways that would be memorable in an interview setting

Context:
{context}
"""

QUIZ_GENERATE_SYSTEM = """You are a senior ML engineer conducting a rigorous technical interview.
Generate ONE challenging, specific interview question based on the provided context.

Requirements:
- The question must be answerable from the provided context
- Target a "medium-hard" difficulty (requires deep understanding, not just recall)
- Do NOT provide the answer
- Do NOT say "Based on the context" or similar meta-references
- Frame it as you would in a real interview
- After the question, add a new line: [TOPIC: <topic_name>]

Context:
{context}
"""

EVALUATE_ANSWER_SYSTEM = """You are a senior ML engineer evaluating a candidate's interview answer.
Grade the candidate's response against the correct information from the provided context.

Structure your evaluation as:
1. **Score**: X/10
2. **What was correct**: List the accurate points made
3. **What was missing or incorrect**: Gaps and errors (be specific)
4. **Model Answer**: A concise, ideal answer the candidate should have given
5. **Key takeaway**: One sentence the candidate should remember

Be honest and technically rigorous. Do not inflate scores.

Interview Question:
{question}

Reference Context:
{context}
"""

HINT_SYSTEM = """You are a helpful interview coach.
Give ONE useful hint for the current interview question WITHOUT giving away the full answer.
The hint should guide the candidate's thinking without spoiling the learning experience.

Question:
{question}

Context (do not reveal directly):
{context}
"""


# ── Agent Class ───────────────────────────────────────────────────────────────

class InterviewAgent:
    """
    LangGraph-based RAG Interview Agent.

    Wraps a compiled StateGraph with helper methods for initialization
    and state-preserving conversation.
    """

    def __init__(self) -> None:
        validate_env()
        self._llm = self._init_llm()
        self._vectorstore = self._init_vectorstore()
        self._graph = self._build_graph()

    # ── Initialization ────────────────────────────────────────────────────────

    def _init_llm(self) -> ChatAnthropic:
        return ChatAnthropic(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
            anthropic_api_key=os.environ["ANTHROPIC_API_KEY"],
        )

    def _init_vectorstore(self) -> Chroma:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        vs = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=CHROMA_PERSIST_DIR,
        )
        count = vs._collection.count()
        if count == 0:
            raise RuntimeError(
                "ChromaDB collection is empty. Run 'python ingest.py' first."
            )
        return vs

    # ── LangGraph Nodes ───────────────────────────────────────────────────────

    def _node_classify_intent(self, state: AgentState) -> AgentState:
        """Determine user intent from the latest message."""
        last_human = next(
            (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
            ""
        )

        has_active_quiz = bool(state.get("current_quiz_question"))

        context_note = (
            "There IS an active quiz question the user may be answering."
            if has_active_quiz
            else "There is NO active quiz question."
        )

        response = self._llm.invoke([
            SystemMessage(content=CLASSIFIER_SYSTEM),
            HumanMessage(content=f"Active quiz context: {context_note}\n\nUser message: {last_human}"),
        ])

        raw = response.content.strip().lower()

        # Safe mapping
        valid_intents: set[Intent] = {"chat", "quiz_request", "quiz_answer", "hint_request"}
        intent: Intent = raw if raw in valid_intents else "chat"

        # Guard: can't answer a quiz if none exists
        if intent == "quiz_answer" and not has_active_quiz:
            intent = "chat"

        return {**state, "intent": intent}

    def _node_retrieve(self, state: AgentState) -> AgentState:
        """Retrieve relevant chunks from ChromaDB based on user message."""
        last_human = next(
            (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
            ""
        )

        # For quiz answers, also incorporate the quiz question text in retrieval
        if state.get("intent") == "quiz_answer" and state.get("current_quiz_question"):
            query = f"{state['current_quiz_question']} {last_human}"
        else:
            query = last_human

        results = self._vectorstore.similarity_search_with_relevance_scores(
            query=query,
            k=TOP_K_RETRIEVAL,
        )

        # Filter low-relevance results (score < 0.3)
        filtered = [(doc, score) for doc, score in results if score >= 0.3]
        if not filtered:
            filtered = results[:2]  # Fallback: take top 2 regardless

        docs = [doc for doc, _ in filtered]
        doc_texts = [doc.page_content for doc in docs]
        doc_metas = [doc.metadata for doc in docs]

        return {**state, "retrieved_docs": doc_texts, "retrieved_metadata": doc_metas}

    def _node_rag_respond(self, state: AgentState) -> AgentState:
        """Generate a grounded answer for conversational queries."""
        last_human = next(
            (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
            ""
        )

        context = "\n\n---\n\n".join(state["retrieved_docs"])

        prompt = ChatPromptTemplate.from_messages([
            ("system", RAG_RESPOND_SYSTEM),
            ("human", "{question}"),
        ])

        chain = prompt | self._llm
        response = chain.invoke({"context": context, "question": last_human})
        answer = response.content

        return {
            **state,
            "messages": [AIMessage(content=answer)],
            "last_response": answer,
        }

    def _node_generate_quiz(self, state: AgentState) -> AgentState:
        """Generate a quiz question from retrieved context."""
        context = "\n\n---\n\n".join(state["retrieved_docs"])

        prompt = ChatPromptTemplate.from_messages([
            ("system", QUIZ_GENERATE_SYSTEM),
            ("human", "Generate a challenging interview question from the provided context."),
        ])

        chain = prompt | self._llm
        response = chain.invoke({"context": context})
        raw_question = response.content

        # Parse topic tag if present
        topic = "Deep Learning"
        question_text = raw_question
        if "[TOPIC:" in raw_question:
            parts = raw_question.rsplit("[TOPIC:", 1)
            question_text = parts[0].strip()
            topic = parts[1].replace("]", "").strip()

        formatted = (
            f"📋 **Interview Question**\n\n"
            f"{question_text}\n\n"
            f"*Topic: {topic} | Type your answer below. Say 'hint' if you need a nudge.*"
        )

        return {
            **state,
            "messages": [AIMessage(content=formatted)],
            "current_quiz_question": question_text,
            "current_quiz_context": context,
            "current_quiz_topic": topic,
            "last_response": formatted,
        }

    def _node_evaluate_answer(self, state: AgentState) -> AgentState:
        """Grade the user's answer to the active quiz question."""
        last_human = next(
            (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
            ""
        )

        question = state.get("current_quiz_question", "")
        # Use both current quiz context and freshly retrieved docs
        quiz_context = state.get("current_quiz_context", "")
        fresh_context = "\n\n---\n\n".join(state["retrieved_docs"])
        combined_context = f"{quiz_context}\n\n---ADDITIONAL CONTEXT---\n\n{fresh_context}"

        prompt = ChatPromptTemplate.from_messages([
            ("system", EVALUATE_ANSWER_SYSTEM),
            ("human", "Candidate's answer: {answer}"),
        ])

        chain = prompt | self._llm
        response = chain.invoke({
            "question": question,
            "context": combined_context,
            "answer": last_human,
        })

        evaluation = response.content
        full_response = f"🎯 **Answer Evaluation**\n\n{evaluation}"

        return {
            **state,
            "messages": [AIMessage(content=full_response)],
            # Clear quiz state after evaluation
            "current_quiz_question": None,
            "current_quiz_context": None,
            "current_quiz_topic": None,
            "last_response": full_response,
        }

    def _node_hint(self, state: AgentState) -> AgentState:
        """Provide a hint for the current quiz question."""
        question = state.get("current_quiz_question", "")
        context = state.get("current_quiz_context", "")

        prompt = ChatPromptTemplate.from_messages([
            ("system", HINT_SYSTEM),
            ("human", "Please give me a hint."),
        ])

        chain = prompt | self._llm
        response = chain.invoke({"question": question, "context": context})
        hint_text = f"💡 **Hint**\n\n{response.content}"

        return {
            **state,
            "messages": [AIMessage(content=hint_text)],
            "last_response": hint_text,
        }

    # ── Routing ───────────────────────────────────────────────────────────────

    def _route_after_classify(self, state: AgentState) -> str:
        """Decide whether to retrieve before the next node."""
        intent = state.get("intent", "chat")
        if intent == "hint_request":
            return "hint"          # Hint uses cached quiz context, no retrieval needed
        return "retrieve"          # All other intents need fresh retrieval

    def _route_after_retrieve(self, state: AgentState) -> str:
        """Route to the correct generation node based on intent."""
        intent = state.get("intent", "chat")
        routing = {
            "chat": "rag_respond",
            "quiz_request": "generate_quiz",
            "quiz_answer": "evaluate_answer",
        }
        return routing.get(intent, "rag_respond")

    # ── Graph Construction ────────────────────────────────────────────────────

    def _build_graph(self):
        """Assemble and compile the LangGraph state machine."""
        builder = StateGraph(AgentState)

        # Register nodes
        builder.add_node("classify_intent", self._node_classify_intent)
        builder.add_node("retrieve", self._node_retrieve)
        builder.add_node("rag_respond", self._node_rag_respond)
        builder.add_node("generate_quiz", self._node_generate_quiz)
        builder.add_node("evaluate_answer", self._node_evaluate_answer)
        builder.add_node("hint", self._node_hint)

        # Define edges
        builder.add_edge(START, "classify_intent")
        builder.add_conditional_edges(
            "classify_intent",
            self._route_after_classify,
            {"retrieve": "retrieve", "hint": "hint"},
        )
        builder.add_conditional_edges(
            "retrieve",
            self._route_after_retrieve,
            {
                "rag_respond": "rag_respond",
                "generate_quiz": "generate_quiz",
                "evaluate_answer": "evaluate_answer",
            },
        )
        builder.add_edge("rag_respond", END)
        builder.add_edge("generate_quiz", END)
        builder.add_edge("evaluate_answer", END)
        builder.add_edge("hint", END)

        return builder.compile()

    # ── Public Interface ──────────────────────────────────────────────────────

    def chat(self, user_message: str, state: AgentState) -> tuple[str, AgentState]:
        """
        Process a user message and return the agent's response + updated state.

        Args:
            user_message: The user's input string.
            state: Current conversation state.

        Returns:
            Tuple of (response_text, new_state).
        """
        updated_state = {
            **state,
            "messages": state["messages"] + [HumanMessage(content=user_message)],
        }

        result = self._graph.invoke(updated_state)
        response = result.get("last_response", "I'm sorry, I encountered an error.")
        return response, result

    def get_initial_state(self) -> AgentState:
        """Return a fresh initial state for a new conversation session."""
        return AgentState(
            messages=[],
            intent=None,
            retrieved_docs=[],
            retrieved_metadata=[],
            current_quiz_question=None,
            current_quiz_context=None,
            current_quiz_topic=None,
            last_response=None,
        )
