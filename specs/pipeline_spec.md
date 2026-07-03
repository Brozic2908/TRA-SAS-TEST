# Đặc tả Luồng Agentic RAG Hải quan bằng LangGraph

## 1. Định nghĩa các Nút trong Đồ thị (Graph Nodes)
- `retrieve_node`: Thực hiện tìm kiếm tương đồng vector trên `pgvector` để lấy tài liệu hải quan liên quan.
- `grade_documents_node`: Đánh giá xem tài liệu lấy ra có thực sự chứa câu trả lời hợp lệ hay không.
- `generate_node`: Tổng hợp câu trả lời pháp lý cuối cùng dựa trên ngữ cảnh hợp lệ.

## 2. Các kịch bản hành vi Đồ thị (LangGraph BDD Scenarios)

Scenario: Thực thi luồng Corrective RAG khi tài liệu nội bộ bị thiếu thông tin
  Given: Người dùng khởi tạo một phiên tra cứu với câu hỏi cụ thể.
  When: Đồ thị bắt đầu chạy từ Điểm khởi đầu (Start) và kích hoạt nút `retrieve_node`.
  Then: Hệ thống chuyển trạng thái sang nút `grade_documents_node` để chấm điểm relevance.
  
  # Cạnh điều kiện 1 (Conditional Edge)
  Condition: Nếu có ít nhất 1 tài liệu đạt điểm chất lượng (> 0.7)
    Then: Đồ thị đi theo nhánh đến nút `generate_node` để sinh câu trả lời.
  
  # Cạnh điều kiện 2 (Không còn nhánh dự phòng)
  Condition: Nếu tất cả tài liệu đều không đạt điểm chất lượng (< 0.7)
    Then: Đồ thị vẫn đi đến nút `generate_node` để sinh câu thông báo không tìm thấy thông tin.

  Then: Nút `generate_node` hoàn thành, hệ thống kiểm tra Hallucination (sự bịa đặt) trước khi kết thúc (End) và trả về JSON qua FastAPI.
  
  # Cạnh điều kiện 3 (Conditional Edge) - Xử lý lỗi hệ thống
  Condition: Xảy ra lỗi kết nối PostgreSQL hoặc lỗi định dạng JSON
    Then: Đồ thị kích hoạt `fallbacks` và gửi thông báo lỗi về cho Admin.
    And: Dừng xử lý thay vì trả về kết quả rỗng.