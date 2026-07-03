# Quy ước Phát triển LangGraph cho Hệ thống RAG (AGENTS.md)

## 1. Định hình cứng Tech Stack & Thư viện (Harness Config)
Hệ thống phải tuân thủ nghiêm ngặt hệ sinh thái công nghệ được định nghĩa dưới đây để đảm bảo tính nhất quán và hiệu năng:
- **Backend:** FastAPI. Bắt buộc lập trình bất đồng bộ (`async`/`await`) để tránh nghẽn Event Loop.
- **Database:** PostgreSQL tích hợp extension `pgvector` phục vụ Similarity Search (sử dụng toán tử Cosine `<=>`).
- **AI Framework:** LangChain kết hợp LangGraph để điều phối và kiểm soát luồng đồ thị.

## 2. Chuẩn hóa Trạng thái Đồ thị (Graph State Schema)
Mọi Nút (Node) trong LangGraph phải tương tác thông qua một State chung được định nghĩa bằng Pydantic hoặc Python TypedDict có Depth = 3. Cấu trúc dữ liệu phải phẳng hoặc rõ ràng dưới dạng YAML để tránh tràn ngập ngữ cảnh (Context Overflow) và tối ưu hóa độ chính xác xử lý:

```yaml
graph_state:
  type: object
  properties:
    question:
      type: string
      description: "Câu hỏi tra cứu gốc của doanh nghiệp"
    documents:
      type: array
      items:
        type: string
      description: "Danh sách các đoạn trích dẫn lấy từ pgvector hoặc Web Search"
    generation:
      type: string
      description: "Câu trả lời cuối cùng do LLM sinh ra"
    search_fallback:
      type: boolean
      description: "Cờ đánh dấu nếu bắt buộc phải kích hoạt nút dự phòng web_search"
  required: ["question", "documents"]
```

## 3. Quy tắc "Phát triển dựa trên Kiểm thử" (Evaluation & Test-Driven)
Quy trình phát triển và kiểm soát chất lượng phải tuân theo kỷ luật Agentic Engineering:
- **Viết Test trước (Test-First Development):** Agent bắt buộc phải viết các ca kiểm thử bằng `pytest` hoặc file JSON đặc tả hành vi (BDD/Gherkin) trước khi triển khai bất kỳ mã logic hệ thống nào.
- **Đánh giá Quỹ đạo (Trajectory Evaluation):** Agent tự chạy đánh giá chuỗi gọi công cụ (tool-calls) theo chế độ `IN_ORDER` hoặc `EXACT` nhằm đảm bảo luồng nghiệp vụ hải quan đi qua đúng các nút kiểm duyệt theo kế hoạch, không bỏ qua các màng lọc an toàn.

## 4. Thiết lập Ranh giới An toàn & Rút gọn Ngữ cảnh (Zero-Trust Guardrails)
Để kiểm soát hành vi và ngăn chặn lỗi từ các tác nhân tự động (Rogue Agent) đối với dữ liệu pháp luật hải quan:
- **Thay đổi cục bộ (Surgical Changes):** Agent chỉ được phép sửa đổi cục bộ, đúng vị trí cần chỉnh sửa. Tuyệt đối không tự ý tái cấu trúc (refactor) bừa bãi hoặc làm thay đổi cấu trúc không liên quan gây ảnh hưởng đến Git branch và tạo ra merge conflict.
- **Ngữ cảnh Động & Kỹ năng (Dynamic Context & Skills):** Không nhồi nhét tài liệu thô vào prompt gốc (gây hiện tượng Context Rot và suy giảm độ chính xác). Thay vào đó, hãy đóng gói các hàm xử lý nghiệp vụ thành các Agent Skills độc lập (`SKILL.md`) chạy trên RAM hoặc gọi qua máy chủ MCP khi có trigger phù hợp.