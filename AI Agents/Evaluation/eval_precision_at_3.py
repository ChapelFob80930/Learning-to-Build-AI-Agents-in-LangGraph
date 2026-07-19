"""
Precision@3 eval for RAG_Agent.py's retriever.

Two-phase script:
  Phase 1 (collect): embeds the PDF, runs a fixed query set directly against
                      the retriever (bypassing the LLM/tool-calling layer),
                      and writes a judging template to relevance_template.json.
  Phase 2 (score):    reads back the template (after you've filled in
                      "relevant": true/false for the top 3 docs of each query)
                      and prints precision@3 per query + the overall average.

Usage:
    python eval_precision_at_3.py collect
    # -> open relevance_template.json, mark relevant true/false for rank 1-3
    python eval_precision_at_3.py score
"""

import os
import sys
import json
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

load_dotenv()

# --- CONFIG: adjust these two paths to match your machine ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(BASE_DIR, "..", "Stock_Market_Performance_2024.pdf")
PERSIST_DIR = os.path.join(BASE_DIR, "chroma_store")
COLLECTION_NAME = "stock_market"
TEMPLATE_PATH = "relevance_template.json"

# --- Fixed query set (15-20 realistic queries about the doc's likely content) ---
# Edit these if they don't match what's actually in your PDF — the point is
# realistic queries a user of this agent would actually ask, not edge cases.
QUERIES = [
    "How did the S&P 500 perform in 2024?",
    "What were the best performing sectors in 2024?",
    "How did tech stocks perform in 2024?",
    "What was the market's reaction to interest rate changes in 2024?",
    "Did the stock market see a recession in 2024?",
    "What was the performance of the Nasdaq in 2024?",
    "How did small-cap stocks perform compared to large-cap in 2024?",
    "What role did AI companies play in 2024 market performance?",
    "What were the worst performing sectors in 2024?",
    "How volatile was the stock market in 2024?",
    "What was the impact of inflation on the 2024 stock market?",
    "How did the Federal Reserve's policy affect markets in 2024?",
    "What were the major market events in 2024?",
    "How did international markets compare to the US market in 2024?",
    "What was the overall market sentiment in 2024?",
    "Which companies had the largest gains in 2024?",
    "How did the bond market compare to equities in 2024?",
    "What were analyst predictions for 2024 market performance?",
    "How did energy stocks perform in 2024?",
    "What was the year-end summary of market performance for 2024?",
]


def build_retriever():
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"PDF file not found: {PDF_PATH}")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    pdf_loader = PyPDFLoader(PDF_PATH)
    pages = pdf_loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    pages_split = text_splitter.split_documents(pages)

    if not os.path.exists(PERSIST_DIR):
        os.makedirs(PERSIST_DIR)

    vectorstore = Chroma.from_documents(
        documents=pages_split,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
        collection_name=COLLECTION_NAME,
    )

    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})


def collect():
    retriever = build_retriever()
    template = []

    for query in QUERIES:
        docs = retriever.invoke(query)
        entry = {"query": query, "results": []}
        for rank, doc in enumerate(docs[:3], start=1):  # only top 3 need judging
            entry["results"].append(
                {
                    "rank": rank,
                    "snippet": doc.page_content[:300].replace("\n", " "),
                    "relevant": None,  # <-- fill this in: true or false
                }
            )
        template.append(entry)

    with open(TEMPLATE_PATH, "w") as f:
        json.dump(template, f, indent=2)

    print(f"Wrote {len(QUERIES)} queries x top-3 results to {TEMPLATE_PATH}")
    print("Open it, read each snippet, set \"relevant\": true or false for each.")
    print("Then run: python eval_precision_at_3.py score")


def score():
    if not os.path.exists(TEMPLATE_PATH):
        raise FileNotFoundError(f"{TEMPLATE_PATH} not found — run 'collect' first.")

    with open(TEMPLATE_PATH) as f:
        template = json.load(f)

    per_query_scores = []
    unjudged = 0

    for entry in template:
        results = entry["results"]
        judged = [r for r in results if r["relevant"] is not None]
        if len(judged) < len(results):
            unjudged += len(results) - len(judged)

        if not judged:
            continue

        relevant_count = sum(1 for r in judged if r["relevant"] is True)
        precision = relevant_count / len(judged)
        per_query_scores.append(precision)
        print(f"{precision:.2f}  {entry['query']}")

    if unjudged:
        print(f"\nWARNING: {unjudged} results still unjudged (relevant: null) — fill these in for an accurate number.")

    if per_query_scores:
        avg = sum(per_query_scores) / len(per_query_scores)
        print(f"\nPrecision@3 (averaged over {len(per_query_scores)} queries): {avg:.4f}")
    else:
        print("No judged results found.")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("collect", "score"):
        print("Usage: python eval_precision_at_3.py [collect|score]")
        sys.exit(1)

    if sys.argv[1] == "collect":
        collect()
    else:
        score()