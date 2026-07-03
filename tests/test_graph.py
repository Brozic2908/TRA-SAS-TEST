import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from langchain_core.runnables import Runnable
from app.core.graph import graph
from app.core.exceptions import DBConnectionError, JSONFormatError, HallucinationError

# Định nghĩa mock classes để giả lập cấu trúc dữ liệu Pydantic trả về từ LLM
class MockDocumentGrade:
    def __init__(self, index: int, relevance_score: float):
        self.index = index
        self.relevance_score = relevance_score

class MockBatchDocumentGrader:
    def __init__(self, grades: list):
        self.grades = grades

class MockHallucinationGrader:
    def __init__(self, is_hallucination: bool, reason: str):
        self.is_hallucination = is_hallucination
        self.reason = reason

# Định nghĩa một Runnable Mock kế thừa từ LangChain Runnable để hoạt động chính xác với toán tử | (pipe)
class MockRunnable(Runnable):
    def __init__(self, return_value):
        self.return_value = return_value

    def invoke(self, input, config=None, **kwargs):
        return self.return_value

    async def ainvoke(self, input, config=None, **kwargs):
        return self.return_value

@pytest.mark.asyncio
async def test_graph_success_trajectory():
    """
    Test kịch bản: Tìm kiếm nội bộ trả về tài liệu hợp lệ -> Đánh giá liên quan đạt điểm cao (>0.7) 
    -> Đi thẳng tới sinh câu trả lời -> Hoàn thành không có bịa đặt.
    Quỹ đạo mong đợi: retrieve -> grade -> generate.
    """
    with patch("app.core.graph.similarity_search", new_callable=AsyncMock) as mock_sim_search, \
         patch("app.core.graph.get_grader_llm") as mock_get_grader, \
         patch("app.core.graph.get_generator_llm") as mock_get_generator:
        
        # 1. Giả lập tìm kiếm trả về 1 tài liệu
        mock_sim_search.return_value = ["Tài liệu hải quan hợp lệ."]
        
        # 2. Giả lập Grader LLM trả về relevance_score = 0.9 (hợp lệ)
        mock_grader_chain = MockRunnable(MockBatchDocumentGrader([MockDocumentGrade(index=0, relevance_score=0.9)]))
        
        # 3. Giả lập Generator LLM trả về câu trả lời sinh ra (Mock dưới dạng Runnable)
        mock_response = MagicMock()
        mock_response.content = "Câu trả lời đúng chuẩn hải quan."
        mock_get_generator.return_value = MockRunnable(mock_response)
        
        # 4. Giả lập Hallucination check: is_hallucination = False
        mock_hal_chain = MockRunnable(MockHallucinationGrader(is_hallucination=False, reason=""))
        
        # Thiết lập side_effect cho hai lần gọi with_structured_output khác nhau
        mock_get_grader.return_value.with_structured_output.side_effect = [mock_grader_chain, mock_hal_chain]
        
        # Khởi chạy đồ thị
        initial_state = {
            "question": "Thủ tục thông quan xuất khẩu",
            "documents": [],
            "generation": "",
            "search_fallback": False
        }
        
        result = await graph.ainvoke(initial_state)
        
        # Xác thực kết quả sinh ra
        assert result["generation"] == "Câu trả lời đúng chuẩn hải quan."
        assert len(result["documents"]) == 1
        assert result["search_fallback"] is False
        mock_sim_search.assert_called_once()

@pytest.mark.asyncio
async def test_graph_fallback_trajectory():
    """
    Test kịch bản: Tìm kiếm nội bộ không có tài liệu liên quan (<0.7) 
    -> Kích hoạt Web Search (Tavily) -> Sinh câu trả lời.
    Quỹ đạo mong đợi: retrieve -> grade -> web_search -> generate.
    """
    with patch("app.core.graph.similarity_search", new_callable=AsyncMock) as mock_sim_search, \
         patch("app.core.graph.web_search", new_callable=AsyncMock) as mock_web_search, \
         patch("app.core.graph.get_grader_llm") as mock_get_grader, \
         patch("app.core.graph.get_generator_llm") as mock_get_generator:
        
        # 1. Giả lập tìm kiếm nội bộ ra tài liệu không liên quan
        mock_sim_search.return_value = ["Tài liệu không liên quan."]
        
        # 2. Giả lập Grader đánh giá relevance_score = 0.2 (thấp hơn 0.7)
        mock_grader_chain = MockRunnable(MockBatchDocumentGrader([MockDocumentGrade(index=0, relevance_score=0.2)]))
        
        # 3. Giả lập Web Search trả về kết quả mới
        mock_web_search.return_value = ["Tài liệu mới từ Web Search."]
        
        # 4. Giả lập Generator LLM sinh câu trả lời (Mock dưới dạng Runnable)
        mock_response = MagicMock()
        mock_response.content = "Câu trả lời từ thông tin Web Search."
        mock_get_generator.return_value = MockRunnable(mock_response)
        
        # 5. Giả lập Hallucination check trả về False (không bịa đặt)
        mock_hal_chain = MockRunnable(MockHallucinationGrader(is_hallucination=False, reason=""))
        
        mock_get_grader.return_value.with_structured_output.side_effect = [mock_grader_chain, mock_hal_chain]
        
        initial_state = {
            "question": "Quy định mới về xuất nhập khẩu chưa cập nhật",
            "documents": [],
            "generation": "",
            "search_fallback": False
        }
        
        result = await graph.ainvoke(initial_state)
        
        assert result["generation"] == "Tôi là Trợ lý AI chuyên ngành Hải quan Việt Nam. Câu hỏi của bạn nằm ngoài phạm vi tài liệu hiện tại. Vui lòng đặt câu hỏi liên quan đến nghiệp vụ hải quan, xuất nhập khẩu hoặc biểu thuế."
        assert result["documents"] == ["Tài liệu mới từ Web Search."]
        assert result["search_fallback"] is True
        mock_web_search.assert_called_once()

@pytest.mark.asyncio
async def test_graph_database_error_handling():
    """
    Test kịch bản: Kết nối database bị lỗi -> Đồ thị bắn DBConnectionError và dừng xử lý.
    """
    with patch("app.core.graph.similarity_search", new_callable=AsyncMock) as mock_sim_search:
        # Bắn lỗi kết nối database
        mock_sim_search.side_effect = Exception("Connection refused by host db:5432")
        
        initial_state = {
            "question": "Câu hỏi bất kỳ",
            "documents": [],
            "generation": "",
            "search_fallback": False
        }
        
        with pytest.raises(DBConnectionError) as exc_info:
            await graph.ainvoke(initial_state)
            
        assert "Lỗi PostgreSQL" in str(exc_info.value)

@pytest.mark.asyncio
async def test_graph_hallucination_error_handling():
    """
    Test kịch bản: Câu trả lời chứa thông tin bịa đặt (Hallucination) 
    -> Đồ thị bắn HallucinationError và dừng xử lý.
    """
    with patch("app.core.graph.similarity_search", new_callable=AsyncMock) as mock_sim_search, \
         patch("app.core.graph.get_grader_llm") as mock_get_grader, \
         patch("app.core.graph.get_generator_llm") as mock_get_generator:
        
        # 1. Tìm kiếm nội bộ thành công
        mock_sim_search.return_value = ["Quy định A quy định về thủ tục nhập khẩu."]
        
        # 2. Đánh giá tài liệu liên quan = 0.95
        mock_grader_chain = MockRunnable(MockBatchDocumentGrader([MockDocumentGrade(index=0, relevance_score=0.95)]))
        
        # 3. LLM sinh câu trả lời bịa đặt (không có trong tài liệu) (Mock dưới dạng Runnable)
        mock_response = MagicMock()
        mock_response.content = "Quy định A cũng cho phép miễn thuế 100% (bịa đặt)."
        mock_get_generator.return_value = MockRunnable(mock_response)
        
        # 4. Kiểm duyệt viên phát hiện Hallucination = True
        mock_hal_chain = MockRunnable(MockHallucinationGrader(
            is_hallucination=True, 
            reason="Thông tin miễn thuế 100% không có trong tài liệu tham khảo."
        ))
        
        mock_get_grader.return_value.with_structured_output.side_effect = [mock_grader_chain, mock_hal_chain]
        
        initial_state = {
            "question": "Thủ tục và thuế suất Quy định A",
            "documents": [],
            "generation": "",
            "search_fallback": False
        }
        
        # Đồ thị phải bắn lỗi HallucinationError
        with pytest.raises(HallucinationError) as exc_info:
            await graph.ainvoke(initial_state)
            
        assert "Phát hiện lỗi bịa đặt thông tin" in str(exc_info.value)
