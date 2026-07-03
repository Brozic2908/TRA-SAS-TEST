import logging
import os
from contextlib import asynccontextmanager
from typing import List
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import select

from app.database.connection import init_db, async_session, CustomsDocument
from app.core.graph import graph
from app.core.exceptions import DBConnectionError, JSONFormatError, HallucinationError
from app.services.vector_store import add_document

# Thiết lập ghi nhật ký
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("customs_rag")

# Định nghĩa dữ liệu mẫu về hải quan Việt Nam để seed tự động
SEED_DOCS = [
    "Quyết định số 233/QĐ-TCHQ ngày 20/02/2026 ban hành Quy chế kiểm tra xác định xuất xứ hàng hóa xuất khẩu, nhập khẩu. Theo đó, việc kiểm tra xuất xứ phải căn cứ trên hồ sơ hải quan, chứng từ chứng nhận xuất xứ hàng hóa (C/O) và kết quả kiểm tra thực tế hàng hóa.",
    "Quyết định số 259/QĐ-TCHQ quy định về quy trình thủ tục hải quan đối với hàng hóa nhập khẩu thương mại. Doanh nghiệp cần khai báo tờ khai hải quan điện tử qua hệ thống VNACCS/VCIS trước khi hàng hóa đến cửa khẩu.",
    "Quyết định số 263/QĐ-TCHQ hướng dẫn phân loại hàng hóa và áp dụng mức thuế đối với hàng hóa xuất nhập khẩu. Việc phân loại phải tuân thủ danh mục hàng hóa xuất khẩu, nhập khẩu Việt Nam và Quy tắc tổng quát giải thích hệ thống HS.",
    "Công văn 2643/TCHQ-GSQL ngày 15/04/2026 về việc tăng cường giám sát hàng hóa quá cảnh và tạm nhập tái xuất. Yêu cầu kiểm tra nghiêm ngặt container hàng, niêm phong hải quan và định vị GPS hành trình di chuyển.",
    "Công điện 871/CĐ-TTg của Thủ tướng Chính phủ về kiểm soát, ngăn chặn tình trạng buôn lậu và gian lận thương mại qua biên giới. Chỉ đạo Bộ Tài chính và Tổng cục Hải quan phối hợp tuần tra chặt chẽ các lối mở tự phát."
]

async def seed_data_if_empty():
    """
    Tự động nạp dữ liệu mẫu vào cơ sở dữ liệu vector nếu bảng trống.
    """
    try:
        async with async_session() as session:
            stmt = select(CustomsDocument).limit(1)
            res = await session.execute(stmt)
            if not res.scalars().first():
                logger.info("Cơ sở dữ liệu trống. Bắt đầu nạp dữ liệu mẫu...")
                for doc_text in SEED_DOCS:
                    await add_document(session, doc_text)
                logger.info("Đã hoàn thành nạp dữ liệu vector mẫu Hải quan.")
    except Exception as e:
        logger.warning(f"Bỏ qua bước seed dữ liệu mẫu (có thể do thiếu GOOGLE_API_KEY khi khởi động): {str(e)}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Khởi tạo db và pgvector extension
    await init_db()
    # Seed dữ liệu mẫu
    await seed_data_if_empty()
    yield

app = FastAPI(
    title="Customs Q&A Agentic RAG API",
    description="FastAPI backend integrating LangGraph and pgvector for Vietnamese Customs regulations Q&A",
    version="1.0.0",
    lifespan=lifespan
)

# Cấu hình CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
async def get_chatbot_ui():
    template_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    if os.path.exists(template_path):
        with open(template_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h3>Chatbot UI Template index.html not found under app/templates/</h3>"

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    question: str
    documents: List[str]
    generation: str
    search_fallback: bool

@app.post("/api/v1/query", response_model=QueryResponse, status_code=status.HTTP_200_OK)
async def query_endpoint(payload: QueryRequest):
    question = payload.question.strip()
    if not question:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="Câu hỏi của doanh nghiệp không được để trống."
        )
        
    try:
        logger.info(f"Nhận yêu cầu tra cứu: {question}")
        # Khởi tạo trạng thái đầu vào cho LangGraph
        initial_state = {
            "question": question,
            "documents": [],
            "generation": "",
            "search_fallback": False
        }
        
        # Thực thi đồ thị LangGraph bất đồng bộ
        result = await graph.ainvoke(initial_state)
        
        return QueryResponse(
            question=result.get("question", question),
            documents=result.get("documents", []),
            generation=result.get("generation", ""),
            search_fallback=result.get("search_fallback", False)
        )
    except DBConnectionError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi cơ sở dữ liệu hệ thống: {str(e)}"
        )
    except JSONFormatError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi định dạng phản hồi từ LLM: {str(e)}"
        )
    except HallucinationError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Phát hiện lỗi bịa đặt thông tin (Hallucination): {str(e)}"
        )
    except Exception as e:
        logger.error(f"Lỗi hệ thống RAG không xác định: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi hệ thống: {str(e)}"
        )
