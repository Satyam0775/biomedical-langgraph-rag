from typing import TypedDict, List, Any
from langgraph.graph import StateGraph, END
from langgraph_pipeline.nodes import semantic_agent, agentic_agent


# ✅ Define explicit state schema
class RAGState(TypedDict):
    query: str
    index: Any
    semantic_results: List[dict]
    agentic_results: List[dict]


def build_graph():
    """
    LangGraph pipeline:
    semantic_agent → agentic_agent → END
    """

    # ✅ Use explicit state schema
    graph = StateGraph(RAGState)

    # Add nodes
    graph.add_node("semantic_agent", semantic_agent)
    graph.add_node("agentic_agent", agentic_agent)

    # Define flow
    graph.set_entry_point("semantic_agent")
    graph.add_edge("semantic_agent", "agentic_agent")
    graph.add_edge("agentic_agent", END)

    return graph.compile()
