import sys
import os
import asyncio

# Thêm thư mục gốc vào PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.ingest import seed_all_data

if __name__ == "__main__":
    print("🚀 Đang khởi chạy Script Nạp Dữ liệu Pháp luật Hải quan...")
    asyncio.run(seed_all_data())
    print("✅ Hoàn tất!")
