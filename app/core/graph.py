import logging
import asyncio
from typing import Dict, Any, List
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END

from app.schemas.state import GraphState
from app.services.llm import get_grader_llm, get_generator_llm
from app.services.vector_store import similarity_search_structured
from app.database.connection import async_session
from app.core.exceptions import DBConnectionError, JSONFormatError, HallucinationError
from app.core.config import settings

logger = logging.getLogger("customs_rag")

class DocumentGrade(BaseModel):
    index: int = Field(description="Chỉ mục của tài liệu tương ứng (0-indexed)")
    relevance_score: float = Field(description="Điểm số liên quan, từ 0.0 (không liên quan) đến 1.0 (hoàn toàn liên quan)")

class BatchDocumentGrader(BaseModel):
    grades: List[DocumentGrade] = Field(description="Danh sách kết quả chấm điểm cho từng tài liệu")

class HallucinationGrader(BaseModel):
    is_hallucination: bool = Field(
        description="True nếu câu trả lời chứa thông tin bịa đặt/không có trong tài liệu, False nếu trung thực hoàn toàn"
    )
    reason: str = Field(description="Lý do chi tiết cho việc đánh giá")

async def notify_admin(message: str):
    logger.error(f"[ADMIN NOTIFICATION ALERT]: {message}")

def is_customs_query(query: str) -> bool:
    query_lower = query.lower()
    keywords = [
        "hải quan", "hai quan", "thuế", "thue", "xuất khẩu", "xuat khau", "nhập khẩu", "nhap khau",
        "tờ khai", "to khai", "thông quan", "thong quan", "quá cảnh", "qua canh", "chứng từ", "chung tu",
        "hồ sơ", "ho so", "biểu thuế", "bieu thue", "mã hs", "ma hs", "hs code", "luật", "luat",
        "nghị định", "nghi dinh", "thông tư", "thong tu", "quyết định", "quyet dinh", "văn bản", "van ban",
        "cửa khẩu", "cua khau", "logistics", "xử phạt", "xu pat", "vnaccs", "vcis",
        "xuất xứ", "xuat xu", "co c/o", "nghĩa vụ", "nghi vu", "khai báo", "khai bao",
        "kiểm tra", "kiem tra", "giấy phép", "giay phep", "quản lý", "quan ly",
        "đối tượng", "doi tuong", "nguyên tắc", "nguyen tac", "điều", "dieu", "khoản", "khoan",
        "quy định", "quy dinh", "pháp luật", "phap luat", "hàng hóa", "hang hoa", "áp dụng", "ap dung"
    ]
    return any(kw in query_lower for kw in keywords)


OFF_TOPIC_RESPONSE = (
    "Xin chào! Tôi là Trợ lý AI chuyên ngành **Hải quan Việt Nam**, được phát triển trên nền tảng công nghệ RAG (Retrieval-Augmented Generation) kết hợp LangGraph + PostgreSQL/pgvector.\n\n"
    "Tôi có thể giúp bạn tra cứu điều khoản pháp luật, quy trình thủ tục, biểu thuế và các vướng mắc trong hoạt động **xuất nhập khẩu, hải quan, logistics**.\n\n"
    "Xin hãy đặt câu hỏi liên quan đến nghiệp vụ hải quan để tôi hỗ trợ chính xác nhất."
)


async def route_question_node(state: GraphState) -> Dict[str, Any]:
    """Guard: chặn sớm câu hỏi off-topic trước khi đến retrieve."""
    question = state.get("question", "")
    if not is_customs_query(question):
        logger.info(f"Câu hỏi off-topic, trả về ngay: '{question}'")
        return {
            "documents": [],
            "citations": [],
            "generation": OFF_TOPIC_RESPONSE,
            "is_off_topic": True
        }
    return {"is_off_topic": False}

async def retrieve_node(state: GraphState) -> Dict[str, Any]:
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
    question = state.get("question")
    documents = state.get("documents", [])
    logger.info(f"Kích hoạt grade_documents_node chấm điểm {len(documents)} tài liệu.")
    
    has_llm_key = bool(settings.GROQ_API_KEY or settings.GOOGLE_API_KEY or settings.OPENROUTER_API_KEY)
    if not has_llm_key or getattr(settings, "OFFLINE_MODE", False):
        relevant_docs = documents
        return {"documents": relevant_docs}

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
            # Timeout 4s cho grader call
            res = await asyncio.wait_for(grader_chain.ainvoke({"documents_list": documents_list, "question": question}), timeout=4.0)
            
            grade_dict = {g.index: g.relevance_score for g in res.grades}
            for i, doc in enumerate(documents):
                score = grade_dict.get(i, 0.0)
                logger.info(f"Đánh giá tài liệu [{i}]: Điểm relevance = {score}")
                if score >= 0.7:
                    relevant_docs.append(doc)
    except Exception as e:
        logger.warning(f"Lỗi/Timeout LLM Grader ({str(e)}), dùng fallback giữ toàn bộ tài liệu gốc.")
        relevant_docs = documents

    # Nếu grader lỗi và fallback giữ toàn bộ, kiểm tra lại bằng keyword
    if len(relevant_docs) == len(documents) and not is_customs_query(question):
        return {"documents": [], "citations": [], "generation": OFF_TOPIC_RESPONSE, "is_off_topic": True}
    return {"documents": relevant_docs if relevant_docs else documents}

async def generate_node(state: GraphState) -> Dict[str, Any]:
    question = state.get("question")
    documents = state.get("documents", [])
    citations = state.get("citations", [])
    
    if state.get("is_off_topic") or state.get("generation"):
        # Đã có generation từ guard trước, giữ nguyên
        return {}


    
    if not documents:
        return {
            "generation": "Không tìm thấy thông tin phù hợp trong bộ quy định thủ tục Hải quan được cung cấp."
        }

    warnings = []
    for c in citations:
        if c.get("status") == "bi_thay_the" or c.get("superseded_by"):
            warnings.append(
                f"⚠️ [CẢNH BÁO PHÁP LÝ]: Trích dẫn từ văn bản **{c.get('law_number')}** đã bị THAY THẾ bởi **{c.get('superseded_by')}**. "
                f"Doanh nghiệp vui lòng áp dụng quy định mới nhất theo {c.get('superseded_by')}."
            )

    warning_text = "\n\n".join(warnings) if warnings else ""

    # Định dạng ngữ cảnh có gắn nhãn rõ ràng [Số hiệu văn bản - Số điều] cho LLM
    formatted_docs = []
    for c, d in zip(citations, documents):
        law_str = f"Văn bản: {c.get('law_number', 'N/A')}, {c.get('article_number', 'N/A')} - {c.get('title', '')}"
        status_str = f" [CẢNH BÁO: Đã bị thay thế bởi {c.get('superseded_by')}]" if c.get('superseded_by') else ""
        formatted_docs.append(f"📌 [{law_str}{status_str}]\n{d.strip()}")
    context_text = "\n\n---\n\n".join(formatted_docs)

    # Hàm tạo câu trả lời dạng Extractive Fallback khi Offline hoặc Gemini gặp lỗi/hết Quota
    def build_extractive_fallback():
        ext_body = []
        for c, d in zip(citations, documents):
            law_info = f"**{c.get('law_number', 'N/A')} - {c.get('article_number', 'N/A')}** ({c.get('title', '')})"
            ext_body.append(f"📌 {law_info}:\n{d.strip()}")
        ans = "Dựa trên bộ văn bản quy định thủ tục Hải quan Việt Nam, thông tin giải đáp cho câu hỏi của bạn như sau:\n\n" + "\n\n".join(ext_body)
        if warning_text:
            ans = f"{warning_text}\n\n{ans}"
        return ans

    has_llm_key = bool(settings.GROQ_API_KEY or settings.GOOGLE_API_KEY or settings.OPENROUTER_API_KEY)
    if not has_llm_key or getattr(settings, "OFFLINE_MODE", False):
        logger.info("Chạy chế độ Offline Response Generator (Extractive RAG) do không có LLM API Key hoặc OFFLINE_MODE=True...")
        return {"generation": build_extractive_fallback()}

    # Online Mode -> LLM Generation với Prompt kiểm soát chặt chẽ
    try:
        logger.info(f"Kích hoạt LLM Generator cho câu hỏi: '{question}'...")
        generator_llm = get_generator_llm()
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Bạn là Chuyên gia Tư vấn Pháp chế Hải quan Việt Nam.\n"
                       "QUY TẮC BẮT BUỘC KHI TRẢ LỜI (STRICT ZERO-HALLUCINATION & CITATION RULES):\n"
                       "1. BÁM SÁT NGỮ CẢNH: Chỉ sử dụng thông tin từ danh sách tài liệu tham khảo được cung cấp bên dưới. Không dùng kiến thức bên ngoài, không tự suy diễn.\n"
                       "2. TRÍCH DẪN RÕ RÀNG: Mọi ý trả lời BẮT BUỘC phải trích dẫn rõ số điều và số hiệu văn bản (Ví dụ: 'Theo Điều 24 Luật Hải quan 54/2014/QH13...').\n"
                       "3. CẢNH BÁO HIỆU LỰC: Nếu tài liệu chứa cảnh báo hết hiệu lực hoặc bị thay thế (ví dụ NĐ 128/2020 bị thay thế bởi 169/2026), bạn BẮT BUỘC phải thông báo cho doanh nghiệp ngay đầu câu trả lời.\n"
                       "4. KHÔNG TÌM THẤY THÔNG TIN: Nếu tài liệu tham khảo KHÔNG chứa câu trả lời cho câu hỏi của doanh nghiệp, bạn phải trả lời nguyên văn: 'Không tìm thấy thông tin phù hợp trong bộ CSDL quy định thủ tục Hải quan Việt Nam.' Tuyệt đối không bịa đặt."),
            ("human", "TÀI LIỆU PHÁP LUẬT THAM KHẢO:\n\n{context_text}\n\nCÂU HỎI DOANH NGHIỆP: {question}")
        ])
        
        generator_chain = prompt | generator_llm
        
        response = await asyncio.wait_for(generator_chain.ainvoke({
            "context_text": context_text,
            "question": question
        }), timeout=20.0)
        
        generation = response.content
        logger.info(f"LLM Generator sinh thành công câu trả lời ({len(generation)} kí tự).")
        if warning_text and warning_text not in generation:
            generation = f"{warning_text}\n\n{generation}"
        return {"generation": generation}
    except Exception as e:
        logger.warning(f"Lỗi/Timeout LLM Generator ({type(e).__name__}: {str(e)}). Tự động chuyển sang Extractive Fallback!")
        return {"generation": build_extractive_fallback()}

async def check_hallucination_node(state: GraphState) -> Dict[str, Any]:
    """Self-RAG: LLM tự kiểm tra câu trả lời có bịa đặt so với tài liệu không."""
    question = state.get("question")
    documents = state.get("documents", [])
    generation = state.get("generation", "")

    # Bỏ qua nếu không có tài liệu hoặc không có API key (offline fallback)
    has_llm_key = bool(settings.GROQ_API_KEY or settings.GOOGLE_API_KEY or settings.OPENROUTER_API_KEY)
    if not documents or not has_llm_key or getattr(settings, "OFFLINE_MODE", False):
        logger.info("Bỏ qua check_hallucination_node (offline hoặc không có tài liệu).")
        return {"hallucination_checked": True}

    try:
        grader_llm = get_grader_llm()
        hal_grader = grader_llm.with_structured_output(HallucinationGrader)

        hal_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "Bạn là kiểm duyệt viên chống bịa đặt (Anti-Hallucination) cho hệ thống tư vấn pháp lý Hải quan. "
             "Nhiệm vụ: kiểm tra xem câu trả lời có chứa thông tin KHÔNG CÓ TRONG tài liệu tham khảo không. "
             "Nếu câu trả lời chỉ dùng thông tin từ tài liệu và thừa nhận không biết khi thiếu dữ liệu → is_hallucination=False. "
             "Nếu câu trả lời thêm thông tin bịa đặt, số liệu, điều luật không có trong tài liệu → is_hallucination=True."),
            ("human",
             "TÀI LIỆU THAM KHẢO:\n{documents}\n\nCÂU HỎI: {question}\n\nCÂU TRẢ LỜI CẦN KIỂM TRA:\n{generation}")
        ])

        hal_chain = hal_prompt | hal_grader
        docs_text = "\n---\n".join(documents)

        result = await asyncio.wait_for(
            hal_chain.ainvoke({"documents": docs_text, "question": question, "generation": generation}),
            timeout=4.0
        )

        if result.is_hallucination:
            logger.error(f"[HALLUCINATION DETECTED]: {result.reason}")
            await notify_admin(f"Hallucination phát hiện cho câu hỏi: '{question}' — Lý do: {result.reason}")
            from app.core.exceptions import HallucinationError
            raise HallucinationError(f"Câu trả lời chứa thông tin bịa đặt: {result.reason}")

        logger.info("check_hallucination_node: Câu trả lời hợp lệ, không có bịa đặt.")
        return {"hallucination_checked": True}

    except Exception as e:
        if "HallucinationError" in type(e).__name__:
            raise
        logger.warning(f"Lỗi/Timeout Hallucination Grader ({str(e)}), bỏ qua và chấp nhận kết quả.")
        return {"hallucination_checked": True}


def decide_after_route(state: GraphState) -> str:
    """Nếu đã có generation (off-topic), nhảy thẳng generate_node (sẽ skip)."""
    if state.get("is_off_topic") is True or len(state.get("generation", "")) > 0:
        return "generate_node"
    return "retrieve_node"


def decide_after_generate(state: GraphState) -> str:
    """Self-RAG routing: Nếu off-topic hoặc không có generation thì END, ngược lại kiểm tra hallucination."""
    if state.get("is_off_topic") is True or not state.get("generation"):
        return END
    return "check_hallucination_node"

workflow = StateGraph(GraphState)

workflow.add_node("route_question_node", route_question_node)
workflow.add_node("retrieve_node", retrieve_node)
workflow.add_node("grade_documents_node", grade_documents_node)
workflow.add_node("generate_node", generate_node)
workflow.add_node("check_hallucination_node", check_hallucination_node)

workflow.add_edge(START, "route_question_node")
workflow.add_conditional_edges(
    "route_question_node",
    decide_after_route,
    {
        "retrieve_node": "retrieve_node",
        "generate_node": "generate_node"
    }
)
workflow.add_edge("retrieve_node", "grade_documents_node")
workflow.add_edge("grade_documents_node", "generate_node")
workflow.add_conditional_edges(
    "generate_node",
    decide_after_generate,
    {
        "check_hallucination_node": "check_hallucination_node",
        END: END
    }
)
workflow.add_edge("check_hallucination_node", END)

graph = workflow.compile()
