# Copilot Instructions for RAG-Powered Deep Learning Interview Preparation Agent

## Overview
This document provides essential guidance for AI coding agents working with the RAG-Powered Deep Learning Interview Preparation Agent. Understanding the architecture, workflows, and conventions of this codebase will enable effective contributions and enhancements.

## Architecture Overview
The system is built using **LangChain**, **LangGraph**, and **ChromaDB**. The architecture consists of:
- **LangGraph Agent**: Manages intent classification, retrieval, quiz generation, and answer evaluation.
- **ChromaDB**: Stores embedded study materials for efficient retrieval.
- **Anthropic Claude**: Utilized for generating responses.

### Key Components
- **`ingest.py`**: Loads, chunks, embeds, and stores study material into ChromaDB.
- **`agent.py`**: Implements the LangGraph state machine for processing user inputs and generating responses.
- **`main.py`**: Provides an interactive CLI for user interaction.
- **`config.py`**: Centralized configuration for models and paths.
- **`study_material/`**: Contains authored deep learning Markdown documents.

## Developer Workflows
### Setup
1. **Install Dependencies**: Run `pip install -r requirements.txt`.
2. **Configure Environment**: Copy `.env.example` to `.env` and add your `ANTHROPIC_API_KEY`.
3. **Ingest Study Material**: Execute `python ingest.py` to load and embed study materials.
4. **Run the Agent**: Start the agent with `python main.py`.

### Interaction Modes
The agent supports three interaction modes:
- **Quiz Me**: Triggered by phrases like "quiz me" or "test me on CNNs".
- **Answer Evaluation**: Users can type their answers after a quiz question.
- **RAG Chat**: Engage in discussions about topics like backpropagation or transformers.

### Special Commands
- **`quiz`**: Request a quiz question.
- **`hint`**: Get a hint for the current quiz question.
- **`clear`**: Clear conversation history.
- **`topics`**: List all study topics.
- **`quit` / `exit`**: Exit the agent.

## Integration Points
- The agent interacts with **ChromaDB** for data storage and retrieval.
- It utilizes **Anthropic Claude** for generating responses based on user queries.

## Project-Specific Conventions
- Study materials are organized in the `study_material/` directory, with each topic covered in separate Markdown files.
- The ingestion process splits documents into chunks for efficient embedding and retrieval.

## Conclusion
This document serves as a foundational guide for AI agents to navigate and contribute to the RAG-Powered Deep Learning Interview Preparation Agent effectively. For further details, refer to the README.md and the source code.