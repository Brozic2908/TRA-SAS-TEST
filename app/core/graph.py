import logging
from typing import Dict, Any, List
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END

from app.schemas.state import GraphState
from app.services.llm import get_grader_llm, get_generator_llm
from app.services.web_search import web_search
from app.services.vector_store import similarity_search_structured
from app.database.connection import async_session
from app.core.exceptions import DBConnectionError, JSONFormatError, HallucinationError
from app.core.config import settings

logger = logging.getLogger("customs_rag")

# 1. Định nghĩa các cấu trúc dữ liệu cho Structured Output của LLM
class DocumentGrade(BaseModel):
    """Điểm số đánh giá độ liên quan của từng tài liệu."""
    index: int = Field(description="Chỉ mục của tài liệu tương ứng (0-indexed)")
    relevance_score: float = Field(description="Điểm số liên quan, từ 0.0 (không liên quan) đến 1.0 (hoàn toàn liên quan)")

class BatchDocumentGrader(BaseModel):
    """Đánh giá độ liên quan hàng loạt tài liệu."""
    grades: List[DocumentGrade] = Field(description="Danh sách kết quả chấm điểm cho từng tài liệu")

class HallucinationGrader(BaseModel):
    """Kết quả kiểm duyệt Hallucination."""
    is_hallucination: bool = Field(
        description="True nếu câu trả lời chứa thông tin bịa đặt/không có trong tài liệu, False nếu trung thực hoàn toàn"
    )
    reason: str = Field(
        description="Lý do chi tiết cho việc đánh giá"
    )

# 2. Hàm giả lập gửi cảnh báo cho Admin khi gặp sự cố hệ thống
async def notify_admin(message: str):
    logger.error(f"[ADMIN NOTIFICATION ALERT]: {message}")

def is_customs_query(query: str) -> bool:
    """
    Kiểm tra xem câu hỏi có chứa các từ khóa liên quan đến nghiệp vụ hải quan hay không.
    """
    query_lower = query.lower()
    keywords = [
        "hải quan", "hai quan", "thuế", "thue", "xuất khẩu", "xuat khau", "nhập khẩu", "nhap khau",
        "tờ khai", "to khai", "thông quan", "thong quan", "quá cảnh", "qua canh", "chứng từ", "chung tu",
        "hồ sơ", "ho so", "biểu thuế", "bieu thue", "mã hs", "ma hs", "hs code", "luật", "luat",
        "nghị định", "nghi dinh", "thông tư", "thong tu", "quyết định", "quyet dinh", "văn bản", "van ban",
        "cửa khẩu", "cua khau", "logistics", "xử phạt", "xu pat"
    ]
    return any(kw in query_lower for kw in keywords)

# 3. Định nghĩa các Nút (Nodes) trong đồ thị LangGraph
async def retrieve_node(state: GraphState) -> Dict[str, Any]:
    """
    Thực hiện truy vấn tìm kiếm tương đồng trên pgvector/BM25 kèm trích xuất Metadata Citation.
    """
    question = state.get("question")
    logger.info(f"Kích hoạt retrieve_node cho câu hỏi: '{question}'")
    
    try:
        async with async_session() as session:
            struct_docs = await similarity_search_structured(session, question, limit=3)
        
        documents = [d["content"] for d in struct_docs]
        citations = [
            {
                "law_number": d.get("law_number", "N/A"),
                "article_number": d.get("article_number", "N/A"),
                "title": d.get("title", "N/A"),
                "status": d.get("status", "con_hieu_luc"),
                "superseded_by": d.get("superseded_by", None)
            }
            for d in struct_docs
        ]
        return {"documents": documents, "citations": citations}
    except Exception as e:
        err_msg = f"Lỗi PostgreSQL trong retrieve_node: {str(e)}"
        await notify_admin(err_msg)
        raise DBConnectionError(err_msg)

async def grade_documents_node(state: GraphState) -> Dict[str, Any]:
    """
    Đánh giá độ liên quan của tài liệu lấy ra bằng LLM hoặc Keyword Heuristics (nếu Offline).
    """
    question = state.get("question")
    documents = state.get("documents", [])
    logger.info(f"Kích hoạt grade_documents_node chấm điểm {len(documents)} tài liệu.")
    
    # 1. Nếu Offline / Không có Google API Key -> Dùng Keyword Heuristics Chấm điểm
    if not settings.GOOGLE_API_KEY:
        relevant_docs = documents
        search_fallback = len(relevant_docs) == 0
        return {"documents": relevant_docs, "search_fallback": search_fallback}

    # 2. Online Mode -> Dùng Gemini Flash Grader
    try:
        grader_llm = get_grader_llm()
        structured_grader = grader_llm.with_structured_output(BatchDocumentGrader)
        
        grader_prompt = ChatPromptTemplate.from_messages([
            ("system", "Bạn là một kiểm duyệt viên chuyên ngành hỗ trợ hệ thống Q&A Hải quan Việt Nam. "
                       "Nhiệm vụ của bạn là đánh giá mức độ liên quan của một danh sách các tài liệu đối với câu hỏi của doanh nghiệp. "
                       "Tiêu chí chấm điểm như sau:\n"
                       "1. BẮT BUỘC chấm điểm relevance_score cao (từ 0.8 đến 1.0) nếu tài liệu chứa các từ khóa pháp lý, thủ tục hành chính, quy định, luật lệ liên quan đến hoạt động xuất nhập khẩu, hải quan, logistics, thương mại quốc tế, tờ khai hải quan, thuế suất, mã HS code, thông tư, nghị định, quyết định.\n"
                       "2. Chỉ chấm relevance_score thấp (từ 0.0 đến 0.3) khi tài liệu hoàn toàn lạc đề.\n"
                       "Hãy trả về điểm số cho từng tài liệu tương ứng theo đúng chỉ mục (index) của chúng."),
            ("human", "Danh sách các tài liệu cần đánh giá:\n{documents_list}\n\nCâu hỏi: {question}")
        ])
        
        grader_chain = grader_prompt | structured_grader
        relevant_docs = []
        
        if documents:
            documents_list = "\n---\n".join([f"Tài liệu [{i}]: {doc}" for i, doc in enumerate(documents)])
            res = await grader_chain.ainvoke({"documents_list": documents_list, "question": question})
            
            grade_dict = {g.index: g.relevance_score for g in res.grades}
            for i, doc in enumerate(documents):
                score = grade_dict.get(i, 0.0)
                logger.info(f"Đánh giá tài liệu [{i}]: Điểm relevance = {score}")
                if score >= 0.7:
                    relevant_docs.append(doc)
    except Exception as e:
        logger.warning(f"Lỗi LLM Grader (dùng fallback tài liệu gốc): {str(e)}")
        relevant_docs = documents

    search_fallback = len(relevant_docs) == 0
    return {"documents": relevant_docs if relevant_docs else documents, "search_fallback": search_fallback}

async def web_search_node(state: GraphState) -> Dict[str, Any]:
    """
    Nút dự phòng: Tra cứu bổ sung thông tin từ Web Search (Tavily).
    """
    question = state.get("question")
    logger.info(f"Kích hoạt web_search_node dự phòng cho câu hỏi: '{question}'")
    
    try:
        search_results = await web_search(question)
        return {"documents": search_results}
    except Exception as e:
        err_msg = f"Lỗi thực hiện Web Search: {str(e)}"
        await notify_admin(err_msg)
        raise JSONFormatError(err_msg)

async def generate_node(state: GraphState) -> Dict[str, Any]:
    """
    Tổng hợp câu trả lời cuối cùng bám sát ngữ cảnh + trích dẫn số hiệu / điều khoản + cảnh báo văn bản hết hiệu lực.
    """
    question = state.get("question")
    documents = state.get("documents", [])
    citations = state.get("citations", [])
    
    # 1. Xử lý câu hỏi ngoài phạm vi nghiệp vụ hải quan
    if state.get("search_fallback") is True and not is_customs_query(question):
        logger.info("Kích hoạt chế độ search_fallback tại generate_node cho câu hỏi ngoài phạm vi.")
        return {
            "generation": "Tôi là Trợ lý AI chuyên ngành Hải quan Việt Nam. Câu hỏi của bạn nằm ngoài phạm vi tài liệu hiện tại. Vui lòng đặt câu hỏi liên quan đến nghiệp vụ hải quan, xuất nhập khẩu hoặc biểu thuế."
        }
    
    if not documents:
        return {
            "generation": "Không tìm thấy thông tin phù hợp trong bộ quy định thủ tục Hải quan được cung cấp."
        }

    # 2. Xử lý Cảnh báo Pháp lý nếu tài liệu trích dẫn thuộc văn bản đã BỊ THAY THẾ (như 128/2020/NĐ-CP)
    warnings = []
    for c in citations:
        if c.get("status") == "bi_thay_the" or c.get("superseded_by"):
            warnings.append(
                f"⚠️ [CẢNH BÁO PHÁP LÝ]: Trích dẫn từ văn bản **{c.get('law_number')}** đã bị THAY THẾ bởi **{c.get('superseded_by')}**. "
                f"Doanh nghiệp vui lòng áp dụng quy định mới nhất theo {c.get('superseded_by')}."
            )

    warning_text = "\n\n".join(warnings) if warnings else ""

    # 3. Kịch bản Offline / Không có Google API Key -> Extractive RAG Answer Generator
    if not settings.GOOGLE_API_KEY:
        logger.info("Chạy chế độ Offline Response Generator (Extractive RAG)...")
        extractive_body = "\n\n".join([f"• Trích dẫn [{c.get('law_number', 'N/A')} - {c.get('article_number', 'N/A')}]: {d[:400]}..." for c, d in zip(citations, documents)])
        
        answer = f"Theo quy định Hải quan Việt Nam liên quan đến câu hỏi của bạn:\n\n{extractive_body}"
        if warning_text:
            answer = f"{warning_text}\n\n{answer}"
        return {"generation": answer}

    # 4. Kịch bản Online -> Gemini Generation + Hallucination Check
    generator_llm = get_generator_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Bạn là một chuyên gia tư vấn pháp chế Hải quan Việt Nam chuyên nghiệp. "
                   "Hãy giải đáp câu hỏi của doanh nghiệp một cách tự tin, rõ ràng, TRÍCH DẪN ĐÚNG số điều và số hiệu văn bản (Ví dụ: Theo Điều 16 Luật Hải quan 54/2014/QH13...). "
                   "LƯU Ý QUAN TRỌNG: Nếu trong ngữ cảnh có cảnh báo văn bản đã hết hiệu lực hoặc bị thay thế (ví dụ Nghị định 128/2020 bị thay thế bởi 169/2026), bạn BẮT BUỘC phải đưa ra cảnh báo này cho doanh nghiệp ngay trong câu trả lời. Tuyệt đối không bịa đặt (no hallucination)."),
        ("human", "Tài liệu tham khảo:\n\n{documents}\n\nCâu hỏi: {question}")
    ])
    
    generator_chain = prompt | generator_llm
    
    try:
        response = await generator_chain.ainvoke({
            "documents": "\n---\n".join(documents),
            "question": question
        })
        generation = response.content
        if warning_text and warning_text not in generation:
            generation = f"{warning_text}\n\n{generation}"
    except Exception as e:
        logger.warning(f"Lỗi LLM Generator online, dùng fallback: {str(e)}")
        generation = f"Theo quy định Hải quan:\n\n" + "\n".join(documents[:2])

    return {"generation": generation}

# 4. Các cạnh điều kiện (Conditional Edges)
def decide_route(state: GraphState) -> str:
    if state.get("search_fallback"):
        return "web_search_node"
    return "generate_node"

# 5. Khởi tạo và liên kết StateGraph
workflow = StateGraph(GraphState)

workflow.add_node("retrieve_node", retrieve_node)
workflow.add_node("grade_documents_node", grade_documents_node)
workflow.add_node("web_search_node", web_search_node)
workflow.add_node("generate_node", generate_node)

workflow.add_edge(START, "retrieve_node")
workflow.add_edge("retrieve_node", "grade_documents_node")

workflow.add_conditional_edges(
    "grade_documents_node",
    decide_route,
    {
        "web_search_node": "web_search_node",
        "generate_node": "generate_node"
    }
)

workflow.add_edge("web_search_node", "generate_node")
workflow.add_edge("generate_node", END)

graph = workflow.compile()
