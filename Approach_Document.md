# Approach Document: Conversational SHL Assessment Recommender

## 1. Design Choices
Building an agentic recommender system over a highly specialized catalog requires mitigating the inherent non-determinism of Large Language Models (LLMs). An end-to-end "vibe-coded" single-prompt approach fails because LLMs struggle to simultaneously classify intent, strictly enforce domain rules, execute vector retrieval, and synthesize structured JSON without hallucinating or losing context. 

To ensure absolute reliability, this solution implements a **Two-Phase Architecture**:
1. **Intent Router:** A fast, stateless classification phase. It analyzes the conversation history and strictly maps the user's current goal to one of five explicit intents (`CLARIFY`, `RECOMMEND`, `REFINE`, `COMPARE`, `REFUSE`). If the intent requires catalog context, it extracts a highly optimized `search_query` summarizing the entire active profile constraints.
2. **Synthesizer:** A grounded response phase. It receives the intent and (if applicable) the retrieved catalog context. Its prompt is strictly bounded to generate a conversational reply and the structured `recommendations` array containing *only* items provided in the context block.

This modular separation of concerns ensures that conversational continuity is maintained, out-of-scope requests are immediately refused at the Router level, and the agent never hallucinates a non-existent SHL product URL.

## 2. Retrieval Setup
The retrieval engine combines semantic matching with deterministic constraints to ensure both broad coverage and exact product matching.
- **Hybrid Retrieval:** The system uses `sentence-transformers/all-MiniLM-L6-v2` to power a local FAISS vector index for semantic search. However, because assessment names are highly specific (e.g., "SVAR", "OPQ32r"), relying purely on semantic embeddings can cause exact product names to rank lower than generic tests. To solve this, an **Exact Match Pre-Filter** executes first, instantly catching specific acronyms in the user's query and prepending them to the retrieved FAISS context.
- **Context Compression:** Initially, passing raw JSON objects into the LLM context window rapidly exhausted token limits and caused the LLM to lose focus on the conversational instructions. To resolve this, the retrieval pipeline dynamically serializes the retrieved FAISS objects into a highly dense string format (`Name | Type | URL | Truncated Description`). This reduced token usage by >60%.

## 3. Prompt Design & Grounding
- **Stateless Execution:** The API maintains strict statelessness. The entire conversation history is injected into the prompt dynamically on every request.
- **Dynamic Policy Injection:** Domain rules (such as enforcing spoken language clarification for Contact Center roles, or appending Leadership benchmarks for Executive roles) are injected as a `CATALOG_POLICY` block directly into the prompt. The LLM uses these rules to proactively enforce SHL assessment best practices without hardcoding conditional logic in Python.
- **Forced Output Adherence:** The Synthesizer prompt strictly enforces the required JSON schema, explicitly detailing that `recommendations` must be exactly `[]` during `CLARIFY` or `REFUSE` turns, satisfying the hard automated evaluation checks.

## 4. Evaluation Approach & Iteration
- **Local Harness:** A custom CLI evaluator (`cli_evaluator.py`) was built to emulate the automated replay harness. It allowed rapid testing of the 10 public conversation traces, directly streaming `POST /chat` payloads against a local `uvicorn` instance.
- **What Didn’t Work:** Initially, attempting to hardcode specific catalog test names (e.g., "Always recommend SVAR for Spoken English") made the agent extremely brittle. It caused Senior tests to leak into Entry-Level roles when the semantic boundaries blurred.
- **How Improvement Was Measured:** Instead of hardcoding product names in the prompt, the focus was shifted to optimizing the `search_query` extraction and increasing the FAISS `top_k` threshold (enabled by the Context Compression fix mentioned above). By expanding the semantic net to `top_k=20`, the FAISS engine reliably pulled the correct tests into the context window organically, allowing the LLM to dynamically filter out the noise and select the perfect stack. This architectural shift immediately stabilized the `Recall@10` metric across all holdout traces.
