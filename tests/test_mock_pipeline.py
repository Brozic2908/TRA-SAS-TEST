import pytest
from typing import Dict, Any, List
from app.schemas.state import GraphState

# Giả lập các hàm logic của Nút (Nodes) trong đồ thị LangGraph
async def mock_retrieve_node(state: GraphState) -> Dict[str, Any]:
    # Giả lập lấy tài liệu từ pgvector. 
    # Nếu câu hỏi chứa "thiếu thông tin", trả về tài liệu rỗng để kích hoạt fallback
    if "thiếu thông tin" in state["question"]:
        return {"documents": []}
    return {"documents": ["Tài liệu hải quan hợp lệ về thủ tục thông quan hàng hóa."]}

async def mock_grade_documents_node(state: GraphState) -> Dict[str, Any]:
    # Đánh giá chất lượng tài liệu. Nếu rỗng, đánh dấu cần search_fallback
    if not state["documents"]:
        return {"search_fallback": True}
    return {"search_fallback": False}

async def mock_web_search_node(state: GraphState) -> Dict[str, Any]:
    # Giả lập tìm kiếm bổ sung trên web
    return {"documents": ["Tài liệu bổ sung từ Web Search về hải quan."]}

async def mock_generate_node(state: GraphState) -> Dict[str, Any]:
    # Giả lập sinh câu trả lời dựa trên tài liệu hiện có
    if not state["documents"]:
        raise ValueError("Không có tài liệu để sinh câu trả lời!")
    return {"generation": "Câu trả lời giả lập dựa trên tài liệu."}

# Định nghĩa hàm kiểm tra cạnh điều kiện (Conditional Edge)
def decide_to_generate(state: GraphState) -> str:
    if state.get("search_fallback"):
        return "web_search"
    return "generate"

@pytest.mark.asyncio
async def test_corrective_rag_trajectory_with_fallback():
    """
    Kịch bản: Thiếu thông tin trong kho tài liệu nội bộ -> Kích hoạt Web Search -> Sinh câu trả lời.
    Đánh giá Quỹ đạo (Trajectory Evaluation): Bắt buộc đi qua [retrieve -> grade -> web_search -> generate].
    """
    # 1. Khởi tạo State ban đầu
    state: GraphState = {
        "question": "Quy trình hải quan mới thiếu thông tin",
        "documents": [],
        "generation": "",
        "search_fallback": False
    }
    
    trajectory: List[str] = []
    
    # 2. Giả lập chạy đồ thị và ghi lại quỹ đạo (Trajectory Tracking)
    # Bước 1: retrieve
    ret_res = await mock_retrieve_node(state)
    state.update(ret_res)
    trajectory.append("retrieve_node")
    
    # Bước 2: grade
    grade_res = await mock_grade_documents_node(state)
    state.update(grade_res)
    trajectory.append("grade_documents_node")
    
    # Bước 3: Đánh giá cạnh điều kiện
    next_step = decide_to_generate(state)
    assert next_step == "web_search", "Phải chọn đi tiếp sang web_search do thiếu tài liệu nội bộ"
    
    # Bước 4: web_search
    web_res = await mock_web_search_node(state)
    state.update(web_res)
    trajectory.append("web_search_node")
    
    # Bước 5: generate
    gen_res = await mock_generate_node(state)
    state.update(gen_res)
    trajectory.append("generate_node")
    
    # 3. Đánh giá Quỹ đạo (Trajectory Evaluation) chế độ EXACT
    expected_trajectory = ["retrieve_node", "grade_documents_node", "web_search_node", "generate_node"]
    assert trajectory == expected_trajectory, f"Quỹ đạo thực tế {trajectory} không khớp với {expected_trajectory}"
    assert "hải quan" in state["documents"][0], "Dữ liệu thu thập sau Web Search phải hợp lệ"
    assert state["generation"] != "", "Phải sinh ra câu trả lời cuối cùng"

@pytest.mark.asyncio
async def test_corrective_rag_trajectory_success():
    """
    Kịch bản: Kho tài liệu có sẵn thông tin đầy đủ -> Sinh câu trả lời ngay.
    Đánh giá Quỹ đạo (Trajectory Evaluation): Bắt buộc đi qua [retrieve -> grade -> generate].
    """
    state: GraphState = {
        "question": "Thủ tục thông quan hàng hóa",
        "documents": [],
        "generation": "",
        "search_fallback": False
    }
    
    trajectory: List[str] = []
    
    # Bước 1: retrieve
    ret_res = await mock_retrieve_node(state)
    state.update(ret_res)
    trajectory.append("retrieve_node")
    
    # Bước 2: grade
    grade_res = await mock_grade_documents_node(state)
    state.update(grade_res)
    trajectory.append("grade_documents_node")
    
    # Bước 3: Đánh giá cạnh điều kiện
    next_step = decide_to_generate(state)
    assert next_step == "generate", "Phải đi thẳng đến sinh câu trả lời"
    
    # Bước 4: generate
    gen_res = await mock_generate_node(state)
    state.update(gen_res)
    trajectory.append("generate_node")
    
    # Đánh giá Quỹ đạo
    expected_trajectory = ["retrieve_node", "grade_documents_node", "generate_node"]
    assert trajectory == expected_trajectory, f"Quỹ đạo thực tế {trajectory} không khớp với {expected_trajectory}"
    assert state["generation"] != "", "Phải sinh ra câu trả lời cuối cùng"
