# 🧬Biomedical LangGraph Retrieval-Augmented Knowledge System

A production-ready **Retrieval-Augmented Retrieval (RAG)** system built on biomedical literature (PubMed), designed to **compare Semantic Chunking and Agentic Chunking** strategies using **LangGraph**, **Pinecone**, **FastAPI**, and **Streamlit**.

---
<img width="183" height="350" alt="Screenshot 2026-01-15 194712" src="https://github.com/user-attachments/assets/58c60145-07f8-44a4-977e-e9f9b1315f8f" />


## 📌 Project Objective

The objective of this project is to:

- Compare **Semantic Chunking vs Agentic Chunking** on large-scale biomedical text (PubMed).
- Use **vector embeddings** for semantic similarity search.
- Implement **multi-agent retrieval** using LangGraph.
- Store and retrieve embeddings efficiently using **Pinecone**.
- Expose the system via **FastAPI**.
- Provide an interactive querying interface using **Streamlit**.

The system is designed to be **scalable, modular, and production-ready**, focusing on **retrieval strategy comparison**, not answer generation.

---

## 🧠 Chunking Strategies

### 1️⃣ Semantic Chunking
- Text is split into **meaningful, context-preserving units**.
- Focuses on **semantic continuity** between sentences.
- Retrieval is based purely on **contextual similarity**.
- Suitable for complex biomedical documents where understanding context is critical.
- Tends to return **broader results with high recall**.

---

### 2️⃣ Agentic Chunking
- Text chunks are designed based on **how downstream agents process information**.
- Each chunk is enriched with metadata such as `agent_type`:
  - `disease_agent`
  - `method_agent`
  - `general_agent`
- Retrieval uses **metadata-aware filtering** at query time.
- Enables **task-focused retrieval** without re-embedding or re-indexing data.

Agentic chunking improves **usability and interpretability**, especially for broad or multi-intent biomedical queries.

> **Note:** Semantic and agentic retrieval may return overlapping results.  
> The distinction lies in **chunk design, metadata enrichment, and task-level usability**, not necessarily different documents.

---

## 🔍 What the Task Expects (Clarification)

The task explicitly requires:

- Different **chunking strategies**
- Clear **design-level distinction**
- Proper **comparison and evaluation**
- A scalable **retrieval architecture**

The task does **not** require:
- Always-different retrieval outputs
- LLM-based answer generation
- Separate Pinecone indices per chunking strategy

---

## 🧪 Embedding Model Selection

- **Model Used:** `sentence-transformers/all-MiniLM-L6-v2`
- **Embedding Dimension:** 384
- **Similarity Metric:** Cosine similarity

### Why MiniLM?
- Lightweight and fast
- Strong semantic representation
- Scales efficiently to large datasets like PubMed
- Cost-effective and suitable for Pinecone serverless/free tiers

> Sentence-BERT variants (e.g., MPNet) may offer marginally higher accuracy, but MiniLM provides a better **speed–accuracy tradeoff** for large-scale retrieval.

---

## 🗄️ Vector Database

- **Database:** Pinecone (Serverless)
- **Index Type:** Dense
- **Metric:** Cosine similarity
- **Stored Data:**
  - Chunk embeddings
  - Metadata (`agent_type`, `source_doc_id`)

---

## 🔗 LangGraph-Based Multi-Agent Retrieval

LangGraph is used to orchestrate retrieval through multiple agents:

### 🔍 Semantic Agent
- Performs unrestricted similarity search
- No metadata filtering
- High recall, context-based retrieval

### 🤖 Agentic Agent
- Uses metadata-aware interpretation
- Task-focused retrieval
- Improves interpretability and downstream usability

This setup enables **direct comparison** of semantic vs agentic retrieval under the same embedding and index configuration.

---

## 📊 Evaluation: Semantic vs Agentic Chunking

Evaluation is performed using:
- **Retrieval latency**
- **Agent-type distribution**
- **Query intent alignment**

### Key Observations:
- Both strategies retrieve semantically relevant chunks due to shared embedding space.
- Agentic chunking improves **task-level clarity** by aligning results with agent roles.
- Latency differences are minimal, with agentic retrieval sometimes faster due to reduced noise.

> Since PubMed does not provide labeled relevance judgments, evaluation focuses on **behavioral and qualitative metrics**, which is standard for large-scale IR systems.

---

## ⚙️ Tech Stack

- **Language:** Python
- **Chunking & Retrieval:** LangGraph
- **Embeddings:** Sentence-Transformers (MiniLM)
- **Vector Store:** Pinecone
- **Backend API:** FastAPI
- **Frontend:** Streamlit
- **Environment Management:** python-dotenv

---

## 🚀 How to Run the Project (Local)

### 1️⃣ Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt

2️⃣ Configure Environment Variables
Create a .env file:
PINECONE_API_KEY=your_api_key
PINECONE_HOST=your_pinecone_host
PINECONE_INDEX_NAME=biomedical-rag

3️⃣ Preprocess Dataset
python preprocessing/preprocess_pubmed.py

4️⃣ Chunk the Data
python chunking/semantic_chunker.py
python chunking/agentic_chunker.py

5️⃣ Embed & Upload to Pinecone
python embeddings/embed_chunks.py

6️⃣ Run Evaluation
python -m evaluation.chunking_comparison

7️⃣ Run FastAPI Backend
uvicorn api.main:app --reload

API Docs:
http://127.0.0.1:8000/docs

8️⃣ Run Streamlit Frontend
streamlit run streamlit_app/app.py

🧠 How to Interpret Results
Specific queries → similar results across both strategies
Broad or mixed queries → clearer differences:
Semantic: high recall, mixed agent types
Agentic: focused, task-aware agent alignment
This behavior highlights the practical advantage of agentic chunking for real-world retrieval systems.

📦 Deployment Notes
Streamlit is used as the front-end interface.
FastAPI serves as the backend retrieval service.
The system is locally deployable and cloud-compatible (e.g., Hugging Face Spaces if required).
Cloud deployment is optional and not mandatory for this task.

🧑‍💻 Author

Satyam Kumar
Data Science & AI Enthusiast
Focus Areas: NLP, RAG Systems, Speech Processing, Generative AI
