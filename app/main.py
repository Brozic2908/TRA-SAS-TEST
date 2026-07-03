import logging
import os
import json
import uuid
from contextlib import asynccontextmanager
from typing import List, Optional
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import select

from app.database.connection import init_db, async_session, ConversationMessage, LegalDocument
from app.core.graph import graph
from app.core.exceptions import DBConnectionError, JSONFormatError, HallucinationError
from app.services.ingest import seed_all_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("customs_rag")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Khởi tạo db và nạp dữ liệu pháp luật Hải quan thực tế
    try:
        await init_db()
        await seed_all_data()
    except Exception as e:
        logger.warning(f"Lỗi khởi tạo / seed data ban đầu: {str(e)}")
    yield

app = FastAPI(
    title="Customs Q&A Agentic RAG API",
    description="FastAPI backend integrating LangGraph, PostgreSQL (SQL), pgvector, and Citation Metadata for Vietnamese Customs regulations Q&A",
    version="2.0.0",
    lifespan=lifespan
)

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

@app.get("/api/v1/health")
async def health_check():
    return {"status": "healthy", "service": "Customs RAG API", "version": "2.0.0"}

class CitationItem(BaseModel):
    law_number: str
    article_number: str
    title: str
    status: str
    superseded_by: Optional[str] = None

class QueryRequest(BaseModel):
    question: str
    session_id: Optional[str] = None

class QueryResponse(BaseModel):
    session_id: str
    question: str
    documents: List[str]
    citations: List[CitationItem]
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
        
    session_id = payload.session_id or str(uuid.uuid4())
    
    try:
        logger.info(f"Session [{session_id}] - Nhận yêu cầu tra cứu: {question}")
        initial_state = {
            "question": question,
            "documents": [],
            "citations": [],
            "generation": "",
            "search_fallback": False
        }
        
        result = await graph.ainvoke(initial_state)
        
        raw_citations = result.get("citations", [])
        citations = [CitationItem(**c) for c in raw_citations]
        generation = result.get("generation", "")
        search_fallback = result.get("search_fallback", False)
        documents = result.get("documents", [])

        # Lưu lịch sử hội thoại vào CSDL PostgreSQL (SQL)
        try:
            async with async_session() as session:
                msg = ConversationMessage(
                    session_id=session_id,
                    question=question,
                    answer=generation,
                    citations_json=json.dumps([c.dict() for c in citations], ensure_ascii=False),
                    search_fallback=search_fallback
                )
                session.add(msg)
                await session.commit()
        except Exception as db_err:
            logger.warning(f"Lỗi ghi nhận lịch sử hội thoại vào SQL DB: {str(db_err)}")

        return QueryResponse(
            session_id=session_id,
            question=result.get("question", question),
            documents=documents,
            citations=citations,
            generation=generation,
            search_fallback=search_fallback
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

@app.get("/api/v1/conversations/{session_id}")
async def get_conversation_history(session_id: str):
    """
    Endpoint lấy danh sách lịch sử hội thoại theo session_id từ SQL DB.
    """
    try:
        async with async_session() as session:
            stmt = select(ConversationMessage).where(ConversationMessage.session_id == session_id).order_by(ConversationMessage.created_at)
            res = await session.execute(stmt)
            messages = res.scalars().all()
            
            history = []
            for m in messages:
                history.append({
                    "id": m.id,
                    "session_id": m.session_id,
                    "question": m.question,
                    "answer": m.answer,
                    "citations": json.loads(m.citations_json) if m.citations_json else [],
                    "search_fallback": m.search_fallback,
                    "created_at": m.created_at.isoformat() if m.created_at else None
                })
            return {"session_id": session_id, "messages": history}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi lấy lịch sử hội thoại: {str(e)}"
        )

@app.get("/api/v1/documents")
async def get_legal_documents_list():
    """
    Endpoint trả về danh mục toàn bộ các văn bản pháp luật hải quan trong CSDL SQL.
    """
    try:
        async with async_session() as session:
            stmt = select(LegalDocument)
            res = await session.execute(stmt)
            docs = res.scalars().all()
            return [
                {
                    "id": d.id,
                    "law_number": d.law_number,
                    "title": d.title,
                    "doc_type": d.doc_type,
                    "issue_date": d.issue_date,
                    "status": d.status,
                    "superseded_by": d.superseded_by
                }
                for d in docs
            ]
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi lấy danh mục văn bản: {str(e)}"
        )
