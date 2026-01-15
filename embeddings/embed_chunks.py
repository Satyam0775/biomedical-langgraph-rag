import json
from sentence_transformers import SentenceTransformer

# Embedding model (fixed as per assignment)
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)

# Input: agentic chunks (final chunks)
INPUT_PATH = "data/processed/agentic_chunks.json"


def load_chunks():
    """
    Load agentic chunks from JSON
    """
    with open(INPUT_PATH, encoding="utf-8") as f:
        return json.load(f)


def embed_chunks():
    """
    Generate embeddings for agentic chunks
    Returns:
        ids: list of chunk IDs
        embeddings: numpy array
        metadata: list of metadata dicts
    """
    chunks = load_chunks()

    texts = [c["text"] for c in chunks]
    ids = [c["chunk_id"] for c in chunks]

    metadata = [
        {
            "agent_type": c["agent_type"],
            "source_doc_id": c["source_doc_id"]
        }
        for c in chunks
    ]

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True
    )

    return ids, embeddings, metadata
