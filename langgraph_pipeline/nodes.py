from embeddings.embed_chunks import model
from vector_store.pinecone_client import query_index


def embed_query(query: str):
    """
    Embed user query using MiniLM
    """
    return model.encode(query).tolist()


def semantic_agent(state: dict):
    """
    Semantic retrieval agent
    - No filtering
    - Broad, meaning-based retrieval
    """
    query = state["query"]
    index = state["index"]

    query_embedding = embed_query(query)

    # 🔍 NO FILTER
    results = query_index(
        index=index,
        query_embedding=query_embedding,
        top_k=5
    )

    matches = results.matches if hasattr(results, "matches") else []

    return {
        "semantic_results": [
            {
                "id": m.id,
                "score": m.score,
                "metadata": m.metadata
            }
            for m in matches
        ]
    }


def agentic_agent(state: dict):
    """
    Agentic retrieval agent
    - Uses metadata filtering
    - Task-focused retrieval
    """
    query = state["query"]
    index = state["index"]

    query_embedding = embed_query(query)

    # 🧠 Decide agent_type dynamically (simple rule-based)
    query_lower = query.lower()

    if any(word in query_lower for word in ["cancer", "disease", "covid", "tumor", "diagnosis"]):
        agent_type = "disease_agent"
    elif any(word in query_lower for word in ["model", "cnn", "transformer", "architecture", "method"]):
        agent_type = "method_agent"
    else:
        agent_type = "general_agent"

    # 🤖 APPLY FILTER
    results = query_index(
        index=index,
        query_embedding=query_embedding,
        top_k=5,
        filter={"agent_type": agent_type}
    )

    matches = results.matches if hasattr(results, "matches") else []

    return {
        "agentic_results": [
            {
                "id": m.id,
                "score": m.score,
                "metadata": m.metadata
            }
            for m in matches
        ]
    }
