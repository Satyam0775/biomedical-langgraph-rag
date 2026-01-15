from fastapi import FastAPI, Query
from langgraph_pipeline.graph import build_graph
from vector_store.pinecone_client import get_pinecone_index

app = FastAPI(title="Biomedical LangGraph RAG")

# One-time setup
index = get_pinecone_index()
graph = build_graph()


@app.post("/query")
def query_rag(query: str = Query(..., description="Biomedical search query")):
    state = {
        "query": query,
        "index": index
    }

    result = graph.invoke(state)

    return {
        "query": query,
        "semantic_results": result.get("semantic_results", []),
        "agentic_results": result.get("agentic_results", [])
    }
