# Đặc tả Luồng Agentic Self-RAG Hải quan bằng LangGraph

## 1. Định nghĩa các Nút trong Đồ thị (Graph Nodes)
- `route_question_node`: Kiểm tra câu hỏi có thuộc phạm vi hải quan / pháp luật không. Nếu off-topic, trả về ngay mà không cần qua CSDL hay LLM.
- `retrieve_node`: Thực hiện tìm kiếm tương đồng vector trên `pgvector` để lấy tài liệu hải quan liên quan.
- `grade_documents_node`: Self-grading — LLM tự đánh giá xem tài liệu lấy ra có thực sự chứa câu trả lời hợp lệ hay không (closed-domain, không tra mạng ngoài).
- `generate_node`: Tổng hợp câu trả lời pháp lý cuối cùng dựa trên ngữ cảnh hợp lệ.
- `check_hallucination_node`: Self-RAG self-check — LLM tự đối chiếu câu trả lời với tài liệu gốc để phát hiện bịa đặt trước khi trả về API.

## 2. Các kịch bản hành vi Đồ thị (LangGraph BDD Scenarios)

Scenario: Thực thi luồng Self-RAG khi tài liệu nội bộ đầy đủ thông tin
  Given: Người dùng khởi tạo một phiên tra cứu với câu hỏi cụ thể liên quan đến hải quan.
  When: Đồ thị bắt đầu chạy từ Điểm khởi đầu (Start) và kích hoạt nút `route_question_node`.
  Then: Hệ thống chuyển trạng thái sang nút `retrieve_node` để tìm tài liệu liên quan.
  Then: Hệ thống chuyển sang nút `grade_documents_node` để tự chấm điểm relevance (self-grading).
  
  # Cạnh điều kiện 1 (Conditional Edge)
  Condition: Nếu có ít nhất 1 tài liệu đạt điểm chất lượng (>= 0.7)
    Then: Đồ thị đi theo nhánh đến nút `generate_node` để sinh câu trả lời.
  
  # Cạnh điều kiện 2 (Không còn nhánh dự phòng ra ngoài)
  Condition: Nếu tất cả tài liệu đều không đạt điểm chất lượng (< 0.7)
    Then: Đồ thị vẫn đi đến nút `generate_node` để sinh câu thông báo không tìm thấy thông tin.
    Note: Không có web search fallback — hệ thống closed-domain, chỉ dùng tài liệu nội bộ.

  Then: Nút `generate_node` hoàn thành, chuyển sang `check_hallucination_node`.

  # Cạnh điều kiện 3 (Self-RAG Hallucination Check)
  Condition: LLM kiểm tra câu trả lời có bịa đặt (is_hallucination)
    If is_hallucination=False: Câu trả lời hợp lệ, hệ thống trả về JSON qua FastAPI (End).
    If is_hallucination=True: Đồ thị kích hoạt `HallucinationError`, gửi cảnh báo Admin và dừng xử lý.
  
  # Cạnh điều kiện 4 (Xử lý lỗi hệ thống)
  Condition: Xảy ra lỗi kết nối PostgreSQL hoặc lỗi định dạng JSON
    Then: Đồ thị kích hoạt `fallbacks` và gửi thông báo lỗi về cho Admin.
    And: Dừng xử lý thay vì trả về kết quả rỗng.