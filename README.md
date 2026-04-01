Presentation recording Link - https://myunt-my.sharepoint.com/:v:/g/personal/murarinalabothu_my_unt_edu/IQDkrhW8Tt53SZ6FuLYzkjosAa5EJhqKyihbUGVHSrNIxWY?nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJTdHJlYW1XZWJBcHAiLCJyZWZlcnJhbFZpZXciOiJTaGFyZURpYWxvZy1MaW5rIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXcifX0%3D&e=8s4Evx


# RAG-Powered Deep Learning Interview Preparation Agent

A production-grade, agentic interview prep system built with **LangChain**, **LangGraph**, and **ChromaDB**. The agent ingests authored deep learning study material, stores it in a persistent vector database, and enables intelligent chat-based Q&A with two modes: **free-form RAG chat** and **structured quiz generation with answer evaluation**.

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────┐
│                    LangGraph Agent                        │
│                                                           │
│  ┌─────────┐    ┌──────────┐    ┌──────────────────────┐ │
│  │ Classify│───▶│ Retrieve │───▶│  generate_quiz /      │ │
│  │ Intent  │    │  (RAG)   │    │  evaluate_answer /    │ │
│  └─────────┘    └──────────┘    │  rag_respond          │ │
│       ▲                         └──────────────────────┘ │
│       │                                    │              │
│  User Input                           Final Reply         │
└──────────────────────────────────────────────────────────┘
         │                                    │
    ┌────▼────┐                         ┌─────▼──────┐
    │ChromaDB │                         │  Anthropic  │
    │ Vectors │                         │   Claude    │
    └─────────┘                         └────────────┘
```

### Key Components

| Component | Role |
|-----------|------|
| `ingest.py` | Loads, chunks, embeds, and stores study material into ChromaDB |
| `agent.py` | LangGraph state machine with intent classification, retrieval, quiz gen, and answer evaluation |
| `main.py` | Interactive CLI with session memory and colored output |
| `config.py` | Centralized configuration for models, paths, and chunking |
| `study_material/` | 7 authored deep learning Markdown documents |

---

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

### 3. Ingest Study Material

```bash
python ingest.py
```

This will:
- Load all 7 Markdown study guides
- Split them into overlapping chunks (512 tokens, 64 overlap)
- Embed using `all-MiniLM-L6-v2` (local, no API needed)
- Persist to `./chroma_db/`

### 4. Run the Agent

```bash
python main.py
```

---

## Usage

Once running, the agent supports three interaction modes detected automatically:

| Intent | Example Trigger Phrases |
|--------|------------------------|
| **Quiz Me** | "quiz me", "give me a question", "test me on CNNs" |
| **Answer Evaluation** | After a quiz question, type your answer |
| **RAG Chat** | "explain backpropagation", "what is dropout?", "how do transformers work?" |

### Special Commands

| Command | Action |
|---------|--------|
| `quiz` | Request a quiz question |
| `hint` | Get a hint for the current quiz question |
| `clear` | Clear conversation history |
| `topics` | List all study topics |
| `quit` / `exit` | Exit the agent |

---

## Study Material Topics

1. **Neural Networks Fundamentals** — Perceptrons, activation functions, forward pass
2. **Backpropagation & Gradients** — Chain rule, vanishing/exploding gradients
3. **Convolutional Neural Networks** — Convolution, pooling, architectures (VGG, ResNet)
4. **RNNs & LSTMs** — Sequential modeling, gating mechanisms, BPTT
5. **Transformers & Attention** — Self-attention, multi-head attention, positional encoding
6. **Optimization Algorithms** — SGD, Adam, learning rate scheduling
7. **Regularization Techniques** — Dropout, batch norm, weight decay, data augmentation

---

## 👋 Find Your Role


| Role              | You Own                                  | Go To                | Name                         |
|------------------|------------------------------------------|----------------------|------------------------------|
| Corpus Architect | `data/corpus/`                           | → Corpus Architect   | Nalabothu,Murari                    |
| Pipeline Engineer| `config.py`, `store.py`, `nodes.py`, `graph.py` | → Pipeline Engineer  | Babu, HARIHARAN                    |
| UX Lead          | `ui/app.py`                              | → UX Lead            | Boddu, Pranathi                    |
| Prompt Engineer  | `prompts.py`                             | → Prompt Engineer    | SUMANTH MALLESH GUTHI                   |
| QA Lead          | `tests/`, `demo script`                  | → QA Lead            | Kasaraneni, Pranay Krishna                  |

## Project Structure

```
rag_interview_agent/
├── README.md
├── requirements.txt
├── .env.example
├── config.py
├── ingest.py
├── agent.py
├── main.py
├── study_material/
│   ├── 01_neural_networks.md
│   ├── 02_backpropagation.md
│   ├── 03_cnn.md
│   ├── 04_rnn_lstm.md
│   ├── 05_transformers.md
│   ├── 06_optimization.md
│   └── 07_regularization.md
└── chroma_db/          ← created after ingest.py
```
