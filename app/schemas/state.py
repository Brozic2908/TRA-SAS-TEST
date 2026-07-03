from typing import TypedDict, List, Dict, Any

class GraphState(TypedDict):
    """
    Trạng thái (State) đồ thị LangGraph điều phối RAG Hải quan.
    """
    question: str
    documents: List[str]
    citations: List[Dict[str, Any]]
    generation: str
    search_fallback: bool
