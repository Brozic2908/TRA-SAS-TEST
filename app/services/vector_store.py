import logging
import asyncio
from typing import List, Dict, Any, Optional
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.database.connection import CustomsDocument, LegalArticle
from app.services.llm import embeddings
from app.services.bm25_retriever import bm25_index
from app.core.config import settings

logger = logging.getLogger("customs_rag")

async def add_document(
    session: AsyncSession, 
    content: str,
    law_number: Optional[str] = None,
    article_number: Optional[str] = None,
    title: Optional[str] = None,
    status: str = "con_hieu_luc",
    superseded_by: Optional[str] = None
) -> CustomsDocument:
    """
    Thêm một tài liệu mới vào pgvector và bảng customs_docs.
    """
    try:
        vector = None
        if settings.GOOGLE_API_KEY:
            try:
                vector = await asyncio.wait_for(embeddings.aembed_query(content), timeout=4.0)
            except Exception as ex:
                logger.warning(f"Không thể tạo embedding online (fallback rỗng): {str(ex)}")
        
        if vector is None:
            vector = [0.0] * 3072

        doc = CustomsDocument(
            content=content,
            law_number=law_number,
            article_number=article_number,
            title=title,
            status=status,
            superseded_by=superseded_by,
            embedding=vector
        )
        session.add(doc)
        await session.commit()
        await session.refresh(doc)
        logger.info(f"Đã lưu tài liệu ID {doc.id} [{law_number} - {article_number}] thành công.")
        return doc
    except Exception as e:
        logger.error(f"Lỗi thêm tài liệu vào DB: {str(e)}")
        await session.rollback()
        raise e

async def similarity_search_structured(session: AsyncSession, query: str, limit: int = 3) -> List[Dict[str, Any]]:
    """
    Tìm kiếm tài liệu tương đồng dựa trên pgvector cosine distance (<=>)
    hoặc fallback BM25 offline nếu không có Google API Key hoặc API bị treo/lỗi quota.
    """
    # 1. Kịch bản Offline / Không có Google API Key -> Dùng BM25 / SQL ILIKE query
    if not settings.GOOGLE_API_KEY or getattr(settings, "OFFLINE_MODE", False):
        logger.info("Chạy chế độ Offline Search (BM25 Lexical query)...")
        stmt = select(CustomsDocument).limit(100)
        res = await session.execute(stmt)
        all_docs = res.scalars().all()

        doc_dicts = [
            {
                "content": d.content,
                "law_number": d.law_number or "N/A",
                "article_number": d.article_number or "N/A",
                "title": d.title or "N/A",
                "status": d.status or "con_hieu_luc",
                "superseded_by": d.superseded_by or ""
            }
            for d in all_docs
        ]

        bm25_index.fit(doc_dicts)
        results = bm25_index.search(query, top_k=limit)
        if not results and doc_dicts:
            results = doc_dicts[:limit]
        return results

    # 2. Kịch bản Online với pgvector (<=> toán tử Cosine) kèm Timeout 3s tránh treo
    try:
        query_vector = await asyncio.wait_for(embeddings.aembed_query(query), timeout=3.0)
        statement = (
            select(CustomsDocument)
            .order_by(CustomsDocument.embedding.cosine_distance(query_vector))
            .limit(limit)
        )
        result = await session.execute(statement)
        rows = list(result.scalars().all())

        documents = []
        for r in rows:
            documents.append({
                "content": r.content,
                "law_number": r.law_number or "N/A",
                "article_number": r.article_number or "N/A",
                "title": r.title or "N/A",
                "status": r.status or "con_hieu_luc",
                "superseded_by": r.superseded_by or ""
            })
        logger.info(f"Tìm kiếm tương đồng pgvector trả về {len(documents)} kết quả.")
        return documents
    except Exception as e:
        logger.warning(f"Lỗi/Timeout Gemini API Vector Search ({str(e)}). Tự động Fallback sang BM25 Offline...")
        # Tự động Fallback sang BM25 nếu API bị hết quota hoặc treo
        stmt = select(CustomsDocument).limit(100)
        res = await session.execute(stmt)
        all_docs = res.scalars().all()

        doc_dicts = [
            {
                "content": d.content,
                "law_number": d.law_number or "N/A",
                "article_number": d.article_number or "N/A",
                "title": d.title or "N/A",
                "status": d.status or "con_hieu_luc",
                "superseded_by": d.superseded_by or ""
            }
            for d in all_docs
        ]
        bm25_index.fit(doc_dicts)
        results = bm25_index.search(query, top_k=limit)
        return results if results else doc_dicts[:limit]

async def similarity_search(session: AsyncSession, query: str, limit: int = 3) -> List[str]:
    struct_docs = await similarity_search_structured(session, query, limit=limit)
    return [d["content"] for d in struct_docs]
