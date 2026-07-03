class DBConnectionError(Exception):
    """Lỗi kết nối cơ sở dữ liệu PostgreSQL."""
    pass

class JSONFormatError(Exception):
    """Lỗi định dạng dữ liệu JSON hoặc lỗi phân tích cú pháp từ LLM."""
    pass

class HallucinationError(Exception):
    """Lỗi phát hiện thông tin bịa đặt (Hallucination) từ mô hình sinh."""
    pass
