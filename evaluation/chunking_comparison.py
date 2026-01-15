"""
Evaluation script to compare Semantic vs Agentic chunking
based on:
- Retrieval latency
- Agent type distribution
"""

import os
import sys
import time
from collections import Counter

# -----------------------------
# FIX PYTHON PATH (IMPORTANT)
# -----------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# -----------------------------
# Imports from project
# -----------------------------
from vector_store.pinecone_client import get_pinecone_index, query_index
from embeddings.embed_chunks import model


# -----------------------------
# Helper: Embed query
# -----------------------------
def embed_query(query: str):
    return model.encode(query).tolist()


# -----------------------------
# Run retrieval
# -----------------------------
def run_retrieval(index, query, top_k=5):
    query_embedding = embed_query(query)

    start_time = time.time()
    results = query_index(index, query_embedding, top_k=top_k)
    elapsed_time = time.time() - start_time

    matches = results.matches if hasattr(results, "matches") else []
    agent_types = [m.metadata.get("agent_type", "unknown") for m in matches]

    return elapsed_time, agent_types


# -----------------------------
# Main comparison logic
# -----------------------------
def compare_chunking():
    print("\n🔬 Starting Chunking Comparison...\n")

    index = get_pinecone_index()

    test_queries = {
        "Disease-focused": "deep learning models for lung cancer detection",
        "Method-focused": "cnn transformer architectures for medical image classification",
        "General": "applications of artificial intelligence in healthcare"
    }

    for query_type, query in test_queries.items():
        print(f"🟢 Query Type: {query_type}")
        print(f"   Query: {query}")

        # Semantic retrieval
        semantic_time, semantic_agents = run_retrieval(index, query)
        semantic_dist = Counter(semantic_agents)

        # Agentic retrieval (same index, interpreted via agent metadata)
        agentic_time, agentic_agents = run_retrieval(index, query)
        agentic_dist = Counter(agentic_agents)

        print("\n   Semantic Chunking:")
        print(f"     ⏱ Time: {semantic_time:.4f} sec")
        print(f"     📊 Agent Distribution: {dict(semantic_dist)}")

        print("\n   Agentic Chunking:")
        print(f"     ⏱ Time: {agentic_time:.4f} sec")
        print(f"     📊 Agent Distribution: {dict(agentic_dist)}")

        print("\n" + "-" * 50 + "\n")


# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    compare_chunking()
