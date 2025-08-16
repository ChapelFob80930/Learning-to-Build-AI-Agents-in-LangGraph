# Learning to Build AI Agents in LangGraph

A comprehensive project collection for building, testing, and learning about modern AI agent architectures with [LangGraph](https://github.com/langchain-ai/langgraph) and [LangChain](https://github.com/langchain-ai/langchain). This repository contains a variety of agent implementations, document retrieval workflows, conversational patterns, and supporting materials.

---

## 📂 Project Structure

- **AI Agents/**
    - `Agent_Bot.py` – Basic conversational agent using LangGraph and GPT-4.
    - `RAG_Agent.py` – RAG (Retrieval-Augmented Generation) agent with PDF parsing, vector storage (ChromaDB), and information retrieval.
    - `ReAct.py` – ReAct agent pattern with tool integration and conditional graph execution.
    - `memory_agent.py` – Persistent memory conversation agent with logging capabilities.
- **Types of graphs/** – Example Jupyter notebooks demonstrating different LangGraph architectures.
- **Extra/** – Additional chatbot demos (e.g., simple and therapist bots).
- `requirements.txt` – All Python dependencies needed for this project.

---

## 🚀 Key Features

- Modular and extensible agent designs with LangGraph
- Conversational AI agents using OpenAI GPT models
- Retrieval-Augmented Generation with ChromaDB and PDF support
- Persistent memory and conversation logging demos
- Ready-to-run Jupyter Notebook examples of graph patterns
- Tool integration (add, subtract, multiply) demo in ReAct pattern

---

## 🛠️ Installation

1. **Clone the repository:**
    ```
    git clone https://github.com/ChapelFob80930/Learning-to-Build-AI-Agents-in-LangGraph.git
    cd Learning-to-Build-AI-Agents-in-LangGraph
    ```

2. **Install dependencies:**
    ```
    pip install -r requirements.txt
    ```

3. **Set up your OpenAI API credentials and, optionally, other required environment variables:**
    - Copy your `.env` file or set as environment variables according to LangChain and LangGraph documentation.

---

## 💡 Usage

Each agent and notebook is designed to be run independently. Example usage:

```
python AI\ Agents/Agent_Bot.py
python AI\ Agents/RAG_Agent.py
```

Or open and run notebooks in the Types of graphs directory.


- For document-based question answering, ensure you have the sample PDFs placed in the correct folder.
- For persistent memory, conversation logs are saved to `logging.txt`.

---

## 📚 References & Further Learning

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangChain Documentation](https://python.langchain.com/)
- [ChromaDB Documentation](https://docs.trychroma.com/)

---

## 🙏 Acknowledgements

This project is inspired by the LangChain and LangGraph communities.

---

## 📃 License

MIT License. See `LICENSE` for full details.


