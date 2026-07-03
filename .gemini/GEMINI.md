# Cấu hình Model Routing cho Gemini

Dự án RAG Hải quan phân chia các tác vụ cho hai dòng mô hình khác nhau để cân bằng giữa chi phí, tốc độ và chất lượng:

*   **Chấm điểm tài liệu (Document Grading):** Sử dụng dòng mô hình **Gemini Flash 2.0** (`gemini-2.0-flash`) để tối ưu hóa thời gian phản hồi.
*   **Tổng hợp câu trả lời (Answer Generation):** Chuyển sang sử dụng dòng mô hình **Gemini Flash 2.0** (`gemini-2.0-flash`) để tránh lỗi giới hạn hạn ngạch (Rate Limit/Quota 429) của gói Free Tier dòng 2.5-flash.

## Cấu hình định tuyến (Model Routing)

```yaml
models:
  grading: gemini-2.0-flash
  generation: gemini-2.0-flash
```
