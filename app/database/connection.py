import logging
from datetime import datetime
from typing import Optional
from sqlalchemy import text, String, Text, DateTime, ForeignKey
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from pgvector.sqlalchemy import Vector
from app.core.config import settings

logger = logging.getLogger("customs_rag")

DATABASE_URL = settings.get_database_url()
engine = create_async_engine(
    DATABASE_URL, 
    echo=False,
    pool_pre_ping=True,
    pool_recycle=300,
    pool_size=10,
    max_overflow=20
)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

class Base(DeclarativeBase):
    pass

class LegalDocument(Base):
    """
    Bảng SQL lưu thông tin Quản lý Văn bản Pháp luật Hải quan
    """
    __tablename__ = "legal_documents"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    law_number: Mapped[str] = mapped_column(String(100), unique=True, index=True, nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    doc_type: Mapped[str] = mapped_column(String(50), nullable=False)
    issue_date: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    status: Mapped[str] = mapped_column(String(50), default="con_hieu_luc")
    superseded_by: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    articles = relationship("LegalArticle", back_populates="document", cascade="all, delete-orphan")

class LegalArticle(Base):
    """
    Bảng SQL lưu Điều / Khoản văn bản pháp luật + Vector Embedding cho RAG
    """
    __tablename__ = "legal_articles"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    document_id: Mapped[Optional[int]] = mapped_column(ForeignKey("legal_documents.id"), nullable=True)
    law_number: Mapped[str] = mapped_column(String(100), index=True, nullable=False)
    article_number: Mapped[str] = mapped_column(String(50), index=True, nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(50), default="con_hieu_luc")
    superseded_by: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    embedding: Mapped[Optional[list]] = mapped_column(Vector(3072), nullable=True)

    document = relationship("LegalDocument", back_populates="articles")

class CustomsDocument(Base):
    """
    Bảng lưu chunk dữ liệu vector pgvector
    """
    __tablename__ = "customs_docs"
    
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    law_number: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    article_number: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    title: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    status: Mapped[Optional[str]] = mapped_column(String(50), default="con_hieu_luc")
    superseded_by: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    embedding: Mapped[list] = mapped_column(Vector(3072), nullable=False)

class ConversationMessage(Base):
    """
    Bảng SQL lưu Lịch sử hội thoại hỏi - đáp
    """
    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(100), index=True, nullable=False)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    citations_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    search_fallback: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

async def init_db():
    """
    Khởi tạo và di cư (Migration) cơ sở dữ liệu
    """
    try:
        async with engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            logger.info("Extension pgvector đã được kích hoạt.")
        
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            logger.info("Đã kiểm tra/tạo bảng SQL thành công.")

        # Tự động Migration thêm cột mới nếu bảng customs_docs cũ đã tồn tại
        async with engine.begin() as conn:
            cols = [
                ("law_number", "VARCHAR(100)"),
                ("article_number", "VARCHAR(50)"),
                ("title", "TEXT"),
                ("status", "VARCHAR(50) DEFAULT 'con_hieu_luc'"),
                ("superseded_by", "VARCHAR(100)")
            ]
            for col_name, col_type in cols:
                try:
                    await conn.execute(text(f"ALTER TABLE customs_docs ADD COLUMN IF NOT EXISTS {col_name} {col_type};"))
                except Exception:
                    pass

            try:
                await conn.execute(text("ALTER TABLE customs_docs ALTER COLUMN embedding TYPE vector(3072);"))
            except Exception:
                pass

    except Exception as e:
        logger.error(f"Lỗi khởi tạo Database: {str(e)}")
        raise e

async def get_db():
    async with async_session() as session:
        try:
            yield session
        finally:
            await session.close()
