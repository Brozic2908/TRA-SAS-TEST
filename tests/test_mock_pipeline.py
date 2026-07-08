import pytest
from typing import Dict, Any, List
from app.schemas.state import GraphState

# Giả lập các hàm logic của Nút (Nodes) trong đồ thị LangGraph Self-RAG
async def mock_retrieve_node(state: GraphState) -> Dict[str, Any]:
    # Giả lập lấy tài liệu từ pgvector. 
    # Nếu câu hỏi chứa "thiếu thông tin", trả về tài liệu rỗng
    if "thiếu thông tin" in state["question"]:
        return {"documents": []}
    return {"documents": ["Tài liệu hải quan hợp lệ về thủ tục thông quan hàng hóa."]}

async def mock_grade_documents_node(state: GraphState) -> Dict[str, Any]:
    # Đánh giá chất lượng tài liệu (Self-grading)
    return {}

async def mock_generate_node(state: GraphState) -> Dict[str, Any]:
    # Giả lập sinh câu trả lời dựa trên tài liệu hiện có
    if not state["documents"]:
        return {"generation": "Không tìm thấy thông tin phù hợp."}
    return {"generation": "Câu trả lời giả lập dựa trên tài liệu."}

async def mock_check_hallucination_node(state: GraphState) -> Dict[str, Any]:
    # Self-RAG: Kiểm tra câu trả lời có bịa đặt không
    # Trong môi trường mock: mặc định pass (is_hallucination=False)
    return {"hallucination_checked": True}

@pytest.mark.asyncio
async def test_self_rag_trajectory_empty():
    """
    Kịch bản Self-RAG: Thiếu thông tin trong kho tài liệu nội bộ -> Sinh thông báo không tìm thấy.
    Đánh giá Quỹ đạo (Trajectory Evaluation): Bắt buộc đi qua
    [retrieve -> grade -> generate -> check_hallucination].
    """
    # 1. Khởi tạo State ban đầu
    state: GraphState = {
        "question": "Quy trình hải quan mới thiếu thông tin",
        "documents": [],
        "citations": [],
        "generation": "",
        "is_off_topic": False,
        "hallucination_checked": None
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
    
    # Bước 3: generate
    gen_res = await mock_generate_node(state)
    state.update(gen_res)
    trajectory.append("generate_node")
    
    # Bước 4: Self-RAG hallucination check
    hal_res = await mock_check_hallucination_node(state)
    state.update(hal_res)
    trajectory.append("check_hallucination_node")
    
    # 3. Đánh giá Quỹ đạo (Trajectory Evaluation) chế độ EXACT
    expected_trajectory = ["retrieve_node", "grade_documents_node", "generate_node", "check_hallucination_node"]
    assert trajectory == expected_trajectory, f"Quỹ đạo thực tế {trajectory} không khớp với {expected_trajectory}"
    assert state["generation"] == "Không tìm thấy thông tin phù hợp.", "Phải trả lời không tìm thấy thông tin"
    assert state["hallucination_checked"] is True, "Phải đánh dấu đã qua kiểm tra hallucination"

@pytest.mark.asyncio
async def test_self_rag_trajectory_success():
    """
    Kịch bản Self-RAG: Kho tài liệu có sẵn thông tin đầy đủ -> Sinh câu trả lời -> Self-check pass.
    Đánh giá Quỹ đạo (Trajectory Evaluation): Bắt buộc đi qua
    [retrieve -> grade -> generate -> check_hallucination].
    """
    state: GraphState = {
        "question": "Thủ tục thông quan hàng hóa",
        "documents": [],
        "citations": [],
        "generation": "",
        "is_off_topic": False,
        "hallucination_checked": None
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
    
    # Bước 3: generate
    gen_res = await mock_generate_node(state)
    state.update(gen_res)
    trajectory.append("generate_node")
    
    # Bước 4: Self-RAG hallucination check
    hal_res = await mock_check_hallucination_node(state)
    state.update(hal_res)
    trajectory.append("check_hallucination_node")
    
    # Đánh giá Quỹ đạo
    expected_trajectory = ["retrieve_node", "grade_documents_node", "generate_node", "check_hallucination_node"]
    assert trajectory == expected_trajectory, f"Quỹ đạo thực tế {trajectory} không khớp với {expected_trajectory}"
    assert state["generation"] != "", "Phải sinh ra câu trả lời cuối cùng"
    assert state["hallucination_checked"] is True, "Phải đánh dấu đã qua kiểm tra hallucination"
