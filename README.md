---
title: SHL Assessment Recommender
emoji: 🏢
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
---
# Conversational SHL Assessment Recommender

An AI-powered conversational agent that helps users find the right SHL Individual Test Solutions. Built with FastAPI, Groq (Llama 3), and FAISS for hybrid vector/exact-match retrieval.
Deployed API Endpoint: <https://lokeshm25-shl-assessment-task.hf.space/>

## Architecture

* **Data Pipeline (`data_pipeline.py`)**: An offline ETL script that parses the SHL product catalog, cleans JSON control characters (`strict=False`), extracts `test_type` metadata, and builds both a dense FAISS vector index and an exact-match lookup dictionary.
* **Agent (`agent.py`)**: A completely stateless dual-stage LLM engine.
  * **Phase 1: Intent Routing**: Analyzes history to determine intent (CLARIFY, RECOMMEND, REFINE, COMPARE, REFUSE).
  * **Phase 2: Hybrid Retrieval**: Combines exact string matching on product abbreviations (e.g., OPQ32r) with semantic FAISS nearest-neighbor searches to ensure robust Recall@10.
  * **Phase 3: Synthesis**: Uses strict guardrails to prevent hallucinations and strictly format output to the required schema.
* **API (`main.py`)**: A high-performance FastAPI service exposing `/health` and `/chat` endpoints.

## Local Setup

1. **Create and activate a virtual environment**:
   ```bash
   python -m venv .venv
   # Windows:
   .\.venv\Scripts\activate
   # Mac/Linux:
   source .venv/bin/activate
   ```
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Configure Environment variables**:
   Create a `.env` file in the root directory and add your Groq API key:
   ```
   GROQ_API_KEY=gsk_your_key_here
   ```
4. **Build the Index**:
   ```bash
   python data_pipeline.py
   ```
   This will generate `shl_catalog.index` and `catalog_metadata.pkl`.

## Running the API

Start the local server:
```bash
uvicorn main:app --reload
```

## Evaluation (Interactive CLI)

You can interactively chat with the agent in your terminal using the built-in CLI evaluator to test edge cases, timeout constraints, and conversational intent handling.

Open a new terminal while the FastAPI server is running and execute:
```bash
python cli_evaluator.py
```

## Deployment

This architecture is optimized for environments like Hugging Face Spaces (16GB RAM). Simply push the repository, configure the `GROQ_API_KEY` secret, and Hugging Face will automatically detect and run the FastAPI server.
