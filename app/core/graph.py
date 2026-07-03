import logging
from typing import Dict, Any, List
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END

from app.schemas.state import GraphState
from app.services.llm import get_grader_llm, get_generator_llm
from app.services.web_search import web_search
from app.services.vector_store import similarity_search
from app.database.connection import async_session
from app.core.exceptions import DBConnectionError, JSONFormatError, HallucinationError

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
        "cửa khẩu", "cua khau", "logistics"
    ]
    return any(kw in query_lower for kw in keywords)

# 3. Định nghĩa các Nút (Nodes) trong đồ thị LangGraph
async def retrieve_node(state: GraphState) -> Dict[str, Any]:
    """
    Thực hiện truy vấn tìm kiếm tương đồng trên pgvector.
    """
    question = state.get("question")
    logger.info(f"Kích hoạt retrieve_node cho câu hỏi: '{question}'")
    
    try:
        async with async_session() as session:
            docs = await similarity_search(session, question, limit=3)
        return {"documents": docs}
    except Exception as e:
        err_msg = f"Lỗi PostgreSQL trong retrieve_node: {str(e)}"
        await notify_admin(err_msg)
        raise DBConnectionError(err_msg)

async def grade_documents_node(state: GraphState) -> Dict[str, Any]:
    """
    Đánh giá độ liên quan của tài liệu lấy ra bằng Gemini Flash (Model Routing).
    """
    question = state.get("question")
    documents = state.get("documents", [])
    logger.info(f"Kích hoạt grade_documents_node chấm điểm {len(documents)} tài liệu.")
    
    grader_llm = get_grader_llm()
    structured_grader = grader_llm.with_structured_output(BatchDocumentGrader)
    
    grader_prompt = ChatPromptTemplate.from_messages([
        ("system", "Bạn là một kiểm duyệt viên chuyên ngành hỗ trợ hệ thống Q&A Hải quan Việt Nam. "
                   "Nhiệm vụ của bạn là đánh giá mức độ liên quan của một danh sách các tài liệu đối với câu hỏi của doanh nghiệp. "
                   "Tiêu chí chấm điểm như sau:\n"
                   "1. BẮT BUỘC chấm điểm relevance_score cao (từ 0.8 đến 1.0) nếu tài liệu chứa các từ khóa pháp lý, thủ tục hành chính, quy định, luật lệ liên quan đến hoạt động xuất nhập khẩu, hải quan, logistics, thương mại quốc tế, tờ khai hải quan, thuế suất, mã HS code, thông tư, nghị định, quyết định.\n"
                   "2. Chỉ chấm relevance_score thấp (từ 0.0 đến 0.3) khi tài liệu hoàn toàn lạc đề, ví dụ như nói về tâm lý học, cuộc sống cá nhân, hoặc các lĩnh vực hoàn toàn không liên quan đến logistics, thủ tục hải quan hay thương mại quốc tế.\n"
                   "Lưu ý quan trọng: Không bác bỏ tài liệu chỉ vì câu hỏi có chứa mốc thời gian cụ thể (ví dụ: năm 2025, năm 2026) mà tài liệu không có mốc thời gian đó, miễn là tài liệu vẫn chứa thông tin nghiệp vụ/thành phần hồ sơ hải quan liên quan.\n"
                   "Hãy trả về điểm số cho từng tài liệu tương ứng theo đúng chỉ mục (index) của chúng."),
        ("human", "Danh sách các tài liệu cần đánh giá:\n{documents_list}\n\nCâu hỏi: {question}")
    ])
    
    grader_chain = grader_prompt | structured_grader
    
    relevant_docs = []
    
    if documents:
        try:
            documents_list = "\n---\n".join([f"Tài liệu [{i}]: {doc}" for i, doc in enumerate(documents)])
            res = await grader_chain.ainvoke({"documents_list": documents_list, "question": question})
            
            # Khởi tạo dict chứa điểm số theo index
            grade_dict = {g.index: g.relevance_score for g in res.grades}
            for i, doc in enumerate(documents):
                score = grade_dict.get(i, 0.0)
                logger.info(f"Đánh giá tài liệu [{i}]: Điểm relevance = {score}")
                if score >= 0.7:
                    relevant_docs.append(doc)
        except Exception as e:
            err_msg = f"Lỗi định dạng JSON hoặc LLM Grader trong grade_documents_node: {str(e)}"
            await notify_admin(err_msg)
            raise JSONFormatError(err_msg)
            
    # Nếu không có tài liệu nào đạt điểm chất lượng (> 0.7), đánh dấu fallback sang Web Search
    search_fallback = len(relevant_docs) == 0
    logger.info(f"Kết quả chấm điểm: Hợp lệ {len(relevant_docs)}/{len(documents)} tài liệu. search_fallback = {search_fallback}")
    
    # Giữ lại các tài liệu hợp lệ, nếu không có tài liệu nào thì giữ lại danh sách gốc để web_search bổ sung
    return {"documents": relevant_docs if relevant_docs else documents, "search_fallback": search_fallback}

async def web_search_node(state: GraphState) -> Dict[str, Any]:
    """
    Nút dự phòng: Tra cứu bổ sung thông tin từ Web Search (Tavily).
    """
    question = state.get("question")
    logger.info(f"Kích hoạt web_search_node dự phòng cho câu hỏi: '{question}'")
    
    try:
        search_results = await web_search(question)
        logger.info(f"Tìm kiếm web hoàn tất. Đã lấy thêm {len(search_results)} nguồn thông tin mới.")
        # Giữ nguyên cờ search_fallback = True để báo hiệu cho generate_node
        return {"documents": search_results}
    except Exception as e:
        err_msg = f"Lỗi thực hiện Web Search: {str(e)}"
        await notify_admin(err_msg)
        raise JSONFormatError(err_msg)

async def generate_node(state: GraphState) -> Dict[str, Any]:
    """
    Tổng hợp câu trả lời cuối cùng hoặc trả về thông báo mặc định nếu không có tài liệu phù hợp.
    """
    question = state.get("question")
    documents = state.get("documents", [])
    
    # Nếu cờ search_fallback là True và câu hỏi hoàn toàn ngoài phạm vi hải quan
    if state.get("search_fallback") is True and not is_customs_query(question):
        logger.info("Kích hoạt chế độ search_fallback tại generate_node cho câu hỏi ngoài phạm vi. Trả về thông báo mặc định hệ thống.")
        return {
            "generation": "Tôi là Trợ lý AI chuyên ngành Hải quan Việt Nam. Câu hỏi của bạn nằm ngoài phạm vi tài liệu hiện tại. Vui lòng đặt câu hỏi liên quan đến nghiệp vụ hải quan, xuất nhập khẩu hoặc biểu thuế."
        }
    
    logger.info(f"Kích hoạt generate_node để tổng hợp câu trả lời từ {len(documents)} nguồn tài liệu.")
    
    if not documents:
        err_msg = "Không có tài liệu nào sẵn có để sinh câu trả lời."
        await notify_admin(err_msg)
        raise ValueError(err_msg)
        
    # Sinh câu trả lời bằng Gemini Pro (Model Routing)
    generator_llm = get_generator_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Bạn là một chuyên gia tư vấn pháp chế Hải quan Việt Nam chuyên nghiệp. "
                   "Hãy giải đáp câu hỏi của doanh nghiệp một cách tự tin, rõ ràng, trích dẫn đúng các điều khoản, quy định pháp luật dựa trên các tài liệu tham khảo được cung cấp. "
                   "LƯU Ý QUAN TRỌNG: Hãy trả lời trực tiếp và tự nhiên như một chuyên gia tư vấn thực thụ. Tuyệt đối TRÁNH sử dụng các mẫu câu rập khuôn, thiếu tự nhiên ở đầu câu trả lời như: 'Theo tài liệu tham khảo được cung cấp', 'Dựa trên các tài liệu tham khảo', 'Dựa trên thông tin được cung cấp', v.v. Hãy đi thẳng vào nội dung giải đáp."),
        ("human", "Tài liệu tham khảo:\n\n{documents}\n\nCâu hỏi: {question}")
    ])
    
    generator_chain = prompt | generator_llm
    
    try:
        response = await generator_chain.ainvoke({
            "documents": "\n---\n".join(documents),
            "question": question
        })
        generation = response.content
    except Exception as e:
        err_msg = f"Lỗi sinh câu trả lời từ LLM: {str(e)}"
        await notify_admin(err_msg)
        raise JSONFormatError(err_msg)
        
    # Kiểm tra Hallucination bằng Gemini Flash (Model Routing)
    grader_llm = get_grader_llm()
    structured_hallucination_grader = grader_llm.with_structured_output(HallucinationGrader)
    
    hallucination_prompt = ChatPromptTemplate.from_messages([
        ("system", "Bạn là kiểm duyệt viên chuyên đánh giá sự bịa đặt (hallucination) trong câu trả lời RAG. "
                   "Kiểm tra xem câu trả lời (generation) có hoàn toàn dựa trên danh sách tài liệu tham khảo hay không. "
                   "Trả về JSON với trường 'is_hallucination' (True nếu phát hiện bịa đặt/không có trong tài liệu, ngược lại False) và trường 'reason'."),
        ("human", "Tài liệu tham khảo:\n\n{documents}\n\nCâu trả lời cần đánh giá:\n\n{generation}")
    ])
    
    hallucination_chain = hallucination_prompt | structured_hallucination_grader
    
    try:
        hal_res = await hallucination_chain.ainvoke({
            "documents": "\n---\n".join(documents),
            "generation": generation
        })
        if hal_res.is_hallucination:
            err_msg = f"Phát hiện lỗi bịa đặt thông tin (Hallucination): {hal_res.reason}"
            await notify_admin(err_msg)
            raise HallucinationError(err_msg)
    except HallucinationError as he:
        raise he
    except Exception as e:
        err_msg = f"Lỗi định dạng khi kiểm tra Hallucination: {str(e)}"
        await notify_admin(err_msg)
        raise JSONFormatError(err_msg)
        
    logger.info("Hoàn thành generate_node và kiểm tra Hallucination thành công.")
    return {"generation": generation}

# 4. Các cạnh điều kiện (Conditional Edges)
def decide_route(state: GraphState) -> str:
    """
    Cạnh điều kiện điều hướng sau khi grade_documents_node.
    """
    if state.get("search_fallback"):
        return "web_search_node"
    return "generate_node"

# 5. Khởi tạo và liên kết StateGraph
workflow = StateGraph(GraphState)

# Thêm các nút
workflow.add_node("retrieve_node", retrieve_node)
workflow.add_node("grade_documents_node", grade_documents_node)
workflow.add_node("web_search_node", web_search_node)
workflow.add_node("generate_node", generate_node)

# Liên kết các cạnh
workflow.add_edge(START, "retrieve_node")
workflow.add_edge("retrieve_node", "grade_documents_node")

# Cạnh điều kiện từ grade_documents_node
workflow.add_conditional_edges(
    "grade_documents_node",
    decide_route,
    {
        "web_search_node": "web_search_node",
        "generate_node": "generate_node"
    }
)

# Quay lại generate_node sau khi tìm kiếm web
workflow.add_edge("web_search_node", "generate_node")
workflow.add_edge("generate_node", END)

# Biên dịch đồ thị (Compile)
graph = workflow.compile()
