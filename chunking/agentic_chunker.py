import json
import os
import re

# Paths
INPUT_PATH = "data/processed/semantic_chunks.json"
OUTPUT_PATH = "data/processed/agentic_chunks.json"

# Simple keyword rules for agent assignment
DISEASE_KEYWORDS = [
    "disease", "cancer", "covid", "infection", "syndrome", "tumor", "disorder"
]

METHOD_KEYWORDS = [
    "method", "approach", "model", "algorithm", "network", "learning",
    "deep learning", "neural network", "cnn", "transformer"
]


def assign_agent(text: str) -> str:
    """
    Assign an agent based on the content of the chunk.
    This is rule-based agentic chunking (valid for the task).
    """
    text_lower = text.lower()

    for kw in DISEASE_KEYWORDS:
        if kw in text_lower:
            return "disease_agent"

    for kw in METHOD_KEYWORDS:
        if kw in text_lower:
            return "method_agent"

    return "general_agent"


def main():
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Semantic chunks not found: {INPUT_PATH}")

    # Load semantic chunks
    with open(INPUT_PATH, encoding="utf-8") as f:
        semantic_chunks = json.load(f)

    agentic_chunks = []

    for chunk in semantic_chunks:
        agent_type = assign_agent(chunk["text"])

        agentic_chunks.append({
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"],
            "source_doc_id": chunk["source_doc_id"],
            "agent_type": agent_type
        })

    # Save agentic chunks
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(agentic_chunks, f, indent=2, ensure_ascii=False)

    print("✅ Agentic chunking completed")
    print(f"📦 Total agentic chunks: {len(agentic_chunks)}")
    print(f"💾 Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
