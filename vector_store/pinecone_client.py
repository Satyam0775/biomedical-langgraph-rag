import os
from dotenv import load_dotenv
from pinecone import Pinecone

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_HOST = os.getenv("PINECONE_HOST")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "biomedical-rag")

EMBEDDING_DIM = 384  # MiniLM-L6-v2


def get_pinecone_index():
    """
    Connect to existing Pinecone index (Serverless)
    """
    if not PINECONE_API_KEY or not PINECONE_HOST:
        raise ValueError("Pinecone API key or host missing")

    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(host=PINECONE_HOST)

    return index


def upsert_embeddings(index, ids, embeddings, metadata=None):
    """
    Upload embeddings to Pinecone
    """
    vectors = []

    for i, emb in enumerate(embeddings):
        meta = metadata[i] if metadata else {}
        vectors.append(
            {
                "id": ids[i],
                "values": emb.tolist(),
                "metadata": meta
            }
        )

    index.upsert(vectors=vectors)


def query_index(index, query_embedding, top_k=5, filter=None):
    """
    Query Pinecone for similar vectors
    Supports optional metadata filtering (agentic retrieval)
    """
    result = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        filter=filter
    )
    return result
