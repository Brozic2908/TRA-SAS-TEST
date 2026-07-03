from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from app.core.config import settings

# Khởi tạo đối tượng Embeddings của Google GenAI sử dụng text-embedding-004
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=settings.GOOGLE_API_KEY
)

def get_grader_llm() -> ChatGoogleGenerativeAI:
    """
    Model Routing: Sử dụng Gemini Flash 2.0 cho tác vụ chấm điểm (tốc độ cao)
    """
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.0,
        google_api_key=settings.GOOGLE_API_KEY,
        max_retries=1
    )

def get_generator_llm() -> ChatGoogleGenerativeAI:
    """
    Model Routing: Sử dụng Gemini Flash 2.0 cho tác vụ sinh câu trả lời (tốc độ cao)
    """
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.3,
        google_api_key=settings.GOOGLE_API_KEY,
        max_retries=1
    )
