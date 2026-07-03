from typing import TypedDict, List

class GraphState(TypedDict):
    """
    Trạng thái (State) đồ thị LangGraph điều phối RAG Hải quan.
    """
    question: str
    documents: List[str]
    generation: str
    search_fallback: bool
