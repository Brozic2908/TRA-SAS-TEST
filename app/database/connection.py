import logging
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from pgvector.sqlalchemy import Vector
from app.core.config import settings

logger = logging.getLogger("customs_rag")

DATABASE_URL = settings.get_database_url()
engine = create_async_engine(DATABASE_URL, echo=False)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

class Base(DeclarativeBase):
    pass

class CustomsDocument(Base):
    __tablename__ = "customs_docs"
    
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    content: Mapped[str] = mapped_column(nullable=False)
    # Định dạng Vector 3072 chiều cho phù hợp với gemini-embedding-001 của Google GenAI
    embedding: Mapped[list] = mapped_column(Vector(3072), nullable=False)

async def init_db():
    """
    Khởi tạo cơ sở dữ liệu:
    1. Kích hoạt extension pgvector
    2. Tạo bảng customs_docs nếu chưa tồn tại
    """
    try:
        # Kích hoạt extension vector trước
        async with engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            logger.info("Extension pgvector đã được kiểm tra/kích hoạt thành công.")
        
        # Tạo bảng
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            logger.info("Đã tạo cấu trúc bảng customs_docs thành công.")
            
        # Tự động di chuyển kích thước cột vector lên 3072 chiều nếu bảng đã tồn tại từ trước
        async with engine.begin() as conn:
            try:
                await conn.execute(text("ALTER TABLE customs_docs ALTER COLUMN embedding TYPE vector(3072);"))
                logger.info("Đã cập nhật kiểu dữ liệu cột embedding lên 3072 chiều thành công.")
            except Exception as ex:
                logger.debug(f"Bỏ qua bước di chuyển cột (bảng mới được tạo hoặc đã khớp chiều): {str(ex)}")
    except Exception as e:
        logger.error(f"Lỗi khởi tạo Database: {str(e)}")
        raise e

async def get_db():
    """
    Dependency cung cấp Session bất đồng bộ cho FastAPI
    """
    async with async_session() as session:
        try:
            yield session
        finally:
            await session.close()
