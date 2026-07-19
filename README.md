# Learning to Build AI Agents in LangGraph

A comprehensive project collection for building, testing, and learning about modern AI agent architectures with [LangGraph](https://github.com/langchain-ai/langgraph) and [LangChain](https://github.com/langchain-ai/langchain). This repository contains a variety of agent implementations, document retrieval workflows, conversational patterns, supporting materials, and a retrieval quality evaluation.

---

## 📂 Project Structure

- **AI Agents/**
    - `Agent_Bot.py` – Basic conversational agent using LangGraph and OpenAI GPT models.
    - `RAG_Agent.py` – RAG (Retrieval-Augmented Generation) agent with PDF parsing, vector storage (ChromaDB), and information retrieval.
    - `ReAct.py` – ReAct agent pattern with tool integration (add, subtract, multiply) and conditional graph execution.
    - `memory_agent.py` – Persistent memory conversation agent with conversation logging.
    - `Drafter.py` – Document drafting agent with tool-based updates, file saving, and ElevenLabs text-to-speech playback.
    - **Evaluation/** – Retrieval quality evaluation for `RAG_Agent.py`.
        - `eval_precision_at_3.py` – Standalone script that queries the retriever directly (bypassing the LLM/tool-calling layer) and computes Precision@3 against a manually-judged relevance set.
        - `relevance_template.json` – 20 test queries with manually judged relevance for the top-3 retrieved chunks per query.
        - **Result: Precision@3 = 0.30** (averaged across 20 queries). See "Evaluation" section below for methodology and how to reproduce.
- **Types of graphs/** – Example Jupyter notebooks demonstrating different LangGraph architectures.
- **Extra/** – Additional chatbot demos (simple chatbot and a therapist/logical-routing bot).
- `requirements.txt` – All Python dependencies needed for this project.

---

## 🚀 Key Features

- Modular and extensible agent designs with LangGraph
- Conversational AI agents using OpenAI GPT models
- Retrieval-Augmented Generation with ChromaDB and PDF support
- Persistent memory and conversation logging demos
- Ready-to-run Jupyter Notebook examples of graph patterns
- Tool integration (add, subtract, multiply) demo in ReAct pattern
- Document drafting agent with text-to-speech output
- **Standalone retrieval evaluation (Precision@3) for the RAG agent, decoupled from the LLM's tool-calling loop**

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
    - Create a `.env` file with your API keys (OpenAI, ElevenLabs if using `Drafter.py`) according to LangChain and LangGraph documentation. `.env` is gitignored and never committed.

---

## 💡 Usage

Each agent and notebook is designed to be run independently. Example usage:

```
python AI\ Agents/Agent_Bot.py
python AI\ Agents/RAG_Agent.py
```

Or open and run notebooks in the Types of graphs directory.

- For document-based question answering (`RAG_Agent.py`), ensure `Stock_Market_Performance_2024.pdf` is present in the `AI Agents/` folder. The vector store is persisted locally to a `chroma_store/` folder (gitignored, regenerated automatically on first run).

---

## 📊 Evaluation

`RAG_Agent.py`'s retrieval quality is measured separately from the agent's LLM/tool-calling loop, since routing test queries through the LLM introduces confounds (query rewriting, the model deciding whether to call the retriever at all, multi-step tool chaining) that make it impossible to isolate retriever quality specifically.

**Methodology:**
1. `eval_precision_at_3.py` calls the retriever directly (`retriever.invoke(query)`) against a fixed set of 20 realistic queries about the source PDF.
2. The top 3 results per query are written to `relevance_template.json` for manual relevance judging (true/false per result).
3. Precision@3 is computed per query (relevant results / 3) and averaged across all 20 queries.

**Result:** Precision@3 = **0.30**

This reflects that several test queries (e.g. bond market, energy sector, Federal Reserve policy) intentionally probed topics outside this specific document's actual coverage, to check the retriever doesn't return false-confident irrelevant results. The document primarily covers S&P 500 and mega-cap tech performance, which is reflected in higher per-query scores for on-topic queries (up to 1.00) and lower scores for out-of-scope ones.

**To reproduce:**
```
cd "AI Agents/Evaluation"
python eval_precision_at_3.py collect   # runs queries, writes relevance_template.json
# manually judge relevance (true/false) for each result in relevance_template.json
python eval_precision_at_3.py score     # computes and prints Precision@3
```

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
