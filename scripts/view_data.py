import sys
import os
import asyncio
from sqlalchemy import select

# Thêm thư mục gốc vào PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database.connection import async_session, LegalDocument, LegalArticle, CustomsDocument

async def view_database_content():
    async with async_session() as session:
        # 1. Xem Bảng LegalDocument
        stmt1 = select(LegalDocument)
        res1 = await session.execute(stmt1)
        docs = res1.scalars().all()
        
        print("\n" + "="*85)
        print("📑 1. BẢNG DỮ LIỆU VĂN BẢN PHÁP LUẬT HẢI QUAN (legal_documents)")
        print("="*85)
        print("{:<4} | {:<16} | {:<10} | {:<12} | {:<15} | {:<20}".format("ID", "Số Hiệu", "Loại", "Ngày Ban Hành", "Trạng Thái", "Thay Thế Bởi"))
        print("-" * 85)
        for d in docs:
            st = "✅ Còn hiệu lực" if d.status == 'con_hieu_luc' else "⚠️ BỊ THAY THẾ"
            sub = d.superseded_by if d.superseded_by else "-"
            print("{:<4} | {:<16} | {:<10} | {:<12} | {:<15} | {:<20}".format(d.id, d.law_number, d.doc_type, str(d.issue_date or '-'), st, sub))

        # 2. Xem Bảng LegalArticle / CustomsDocument
        stmt2 = select(CustomsDocument)
        res2 = await session.execute(stmt2)
        articles = res2.scalars().all()

        print("\n" + "="*85)
        print(f"📜 2. BẢNG CHI TIẾT ĐIỀU / KHOẢN & VECTOR EMBEDDING ({len(articles)} Điều khoản)")
        print("="*85)
        for idx, a in enumerate(articles, 1):
            print(f"\n--- [Điều khoản {idx}] ---")
            print(f"• Số hiệu: {a.law_number} | {a.article_number}: {a.title}")
            print(f"• Trạng thái: {a.status} (Thay thế bởi: {a.superseded_by or 'Khổng'})")
            print(f"• Nội dung trích đoạn:\n  {a.content[:250].strip()}...")
        print("\n" + "="*85)

if __name__ == "__main__":
    asyncio.run(view_database_content())
