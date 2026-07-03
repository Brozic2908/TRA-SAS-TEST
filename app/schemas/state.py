from typing import TypedDict, List, Dict, Any, Optional

class GraphState(TypedDict):
    """
    Trạng thái (State) đồ thị LangGraph điều phối RAG Hải quan.
    """
    question: str
    documents: List[str]
    citations: List[Dict[str, Any]]
    generation: str
    is_off_topic: Optional[bool]  # True nếu câu hỏi ngoài phạm vi hải quan
