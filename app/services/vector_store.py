import logging
from typing import List
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.database.connection import CustomsDocument
from app.services.llm import embeddings

logger = logging.getLogger("customs_rag")

async def add_document(session: AsyncSession, content: str) -> CustomsDocument:
    """
    Thêm một tài liệu mới vào pgvector. Tính toán vector embedding và lưu trữ.
    """
    try:
        # Tính toán embedding bất đồng bộ
        vector = await embeddings.aembed_query(content)
        doc = CustomsDocument(content=content, embedding=vector)
        session.add(doc)
        await session.commit()
        await session.refresh(doc)
        logger.info(f"Đã lưu tài liệu ID {doc.id} thành công với vector size {len(vector)}")
        return doc
    except Exception as e:
        logger.error(f"Lỗi thêm tài liệu vào DB: {str(e)}")
        await session.rollback()
        raise e

async def similarity_search(session: AsyncSession, query: str, limit: int = 3) -> List[str]:
    """
    Tìm kiếm tài liệu tương đồng dựa trên toán tử cosine (<=>) trong pgvector.
    """
    try:
        query_vector = await embeddings.aembed_query(query)
        # Sử dụng cosine_distance trong pgvector để tìm kiếm khoảng cách Cosine (toán tử <=>)
        statement = (
            select(CustomsDocument.content)
            .order_by(CustomsDocument.embedding.cosine_distance(query_vector))
            .limit(limit)
        )
        result = await session.execute(statement)
        documents = list(result.scalars().all())
        logger.info(f"Tìm kiếm tương đồng cho truy vấn '{query}' trả về {len(documents)} tài liệu.")
        return documents
    except Exception as e:
        logger.error(f"Lỗi tìm kiếm tương đồng pgvector: {str(e)}")
        raise e
