import os
import re
import zipfile
import xml.etree.ElementTree as ET
import asyncio
import logging
from sqlalchemy import select

from app.database.connection import init_db, async_session, LegalDocument, LegalArticle, CustomsDocument
from app.services.vector_store import add_document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("customs_ingest")

DOC_CATALOG = [
    {
        "law_number": "54/2014/QH13",
        "title": "Luật Hải quan 2014 (Hợp nhất 12/VBHN-VPQH)",
        "doc_type": "Luật",
        "issue_date": "23/06/2014",
        "status": "con_hieu_luc",
        "superseded_by": None
    },
    {
        "law_number": "08/2015/NĐ-CP",
        "title": "Nghị định quy định chi tiết và biện pháp thi hành Luật Hải quan (sửa đổi 59/2018, 167/2025)",
        "doc_type": "Nghị định",
        "issue_date": "21/01/2015",
        "status": "con_hieu_luc",
        "superseded_by": None
    },
    {
        "law_number": "01/2015/NĐ-CP",
        "title": "Nghị định quy định chi tiết thi hành Luật Hải quan về địa bàn hoạt động hải quan (sửa đổi 12/2018, 153/2026)",
        "doc_type": "Nghị định",
        "issue_date": "06/01/2015",
        "status": "con_hieu_luc",
        "superseded_by": None
    },
    {
        "law_number": "128/2020/NĐ-CP",
        "title": "Nghị định quy định xử phạt vi phạm hành chính trong lĩnh vực hải quan",
        "doc_type": "Nghị định",
        "issue_date": "19/10/2020",
        "status": "bi_thay_the",
        "superseded_by": "169/2026/NĐ-CP"
    },
    {
        "law_number": "169/2026/NĐ-CP",
        "title": "Nghị định quy định xử phạt vi phạm hành chính trong lĩnh vực hải quan (Thay thế 128/2020/NĐ-CP)",
        "doc_type": "Nghị định",
        "issue_date": "15/01/2026",
        "status": "con_hieu_luc",
        "superseded_by": None
    },
    {
        "law_number": "68/2016/NĐ-CP",
        "title": "Nghị định quy định điều kiện kinh doanh hàng miễn thuế, kho bãi, địa điểm làm thủ tục hải quan",
        "doc_type": "Nghị định",
        "issue_date": "01/07/2016",
        "status": "con_hieu_luc",
        "superseded_by": None
    },
    {
        "law_number": "38/2015/TT-BTC",
        "title": "Thông tư quy định về thủ tục hải quan; kiểm tra, giám sát hải quan (sửa đổi 39/2018/TT-BTC)",
        "doc_type": "Thông tư",
        "issue_date": "25/03/2015",
        "status": "con_hieu_luc",
        "superseded_by": None
    },
    {
        "law_number": "121/2025/TT-BTC",
        "title": "Thông tư hướng dẫn về quy trình thủ tục hải quan điện tử và kiểm tra sau thông quan",
        "doc_type": "Thông tư",
        "issue_date": "10/02/2025",
        "status": "con_hieu_luc",
        "superseded_by": None
    },
    {
        "law_number": "12/2015/TT-BTC",
        "title": "Thông tư quy định chi tiết về thủ tục cấp chứng chỉ nghiệp vụ khai hải quan (sửa đổi 22/2019)",
        "doc_type": "Thông tư",
        "issue_date": "30/01/2015",
        "status": "con_hieu_luc",
        "superseded_by": None
    },
    {
        "law_number": "33/2023/TT-BTC",
        "title": "Thông tư quy định về xác định xuất xứ hàng hóa xuất khẩu, nhập khẩu",
        "doc_type": "Thông tư",
        "issue_date": "31/05/2023",
        "status": "con_hieu_luc",
        "superseded_by": None
    },
    {
        "law_number": "31/2022/TT-BTC",
        "title": "Thông tư ban hành Danh mục hàng hóa xuất khẩu, nhập khẩu Việt Nam",
        "doc_type": "Thông tư",
        "issue_date": "08/06/2022",
        "status": "con_hieu_luc",
        "superseded_by": None
    }
]

PROCESSED_ARTICLES = [
    {
        "law_number": "54/2014/QH13",
        "article_number": "Điều 16",
        "title": "Địa điểm làm thủ tục hải quan",
        "content": "Điều 16. Địa điểm làm thủ tục hải quan\n1. Địa điểm làm thủ tục hải quan là nơi cơ quan hải quan tiếp nhận, đăng ký và kiểm tra hồ sơ hải quan, kiểm tra thực tế hàng hóa, phương tiện vận tải.\n2. Địa điểm tiếp nhận, đăng ký và kiểm tra hồ sơ hải quan là trụ sở Chi cục Hải quan, Chi cục Hải quan cửa khẩu.\n3. Địa điểm kiểm tra thực tế hàng hóa bao gồm:\na) Địa điểm kiểm tra tại Chi cục Hải quan cửa khẩu;\nb) Địa điểm kiểm tra tập trung theo quyết định của Tổng cục trưởng Tổng cục Hải quan;\nc) Địa điểm kiểm tra tại cơ sở sản xuất, công trình; nơi diễn ra hội chợ, triển lãm;\nd) Địa điểm kiểm tra tại kho ngoại quan, kho bảo thuế, địa điểm gom hàng lẻ;\nđ) Địa điểm kiểm tra khác do Tổng cục trưởng Tổng cục Hải quan quyết định.",
        "status": "con_hieu_luc",
        "superseded_by": None
    },
    {
        "law_number": "54/2014/QH13",
        "article_number": "Điều 24",
        "title": "Hồ sơ hải quan",
        "content": "Điều 24. Hồ sơ hải quan\n1. Hồ sơ hải quan bao gồm:\na) Tờ khai hải quan hoặc chứng từ thay thế tờ khai hải quan;\nb) Chứng từ có liên quan.\n2. Tùy từng trường hợp, người khai hải quan phải nộp hoặc xuất trình hợp đồng mua bán hàng hóa, hóa đơn thương mại, chứng từ vận tải, chứng từ chứng nhận xuất xứ hàng hóa, giấy phép xuất khẩu, nhập khẩu, văn bản thông báo kết quả kiểm tra hoặc miễn kiểm tra chuyên ngành.",
        "status": "con_hieu_luc",
        "superseded_by": None
    },
    {
        "law_number": "54/2014/QH13",
        "article_number": "Điều 29",
        "title": "Khai hải quan",
        "content": "Điều 29. Khai hải quan\n1. Khai hải quan được thực hiện theo phương thức điện tử. Người khai hải quan thực hiện khai hải quan giấy trong các trường hợp theo quy định của Chính phủ.\n2. Tờ khai hải quan đã đăng ký có giá trị làm thủ tục hải quan. Thời hạn hiệu lực của tờ khai hải quan là 15 ngày kể từ ngày đăng ký.\n3. Người khai hải quan được khai bổ sung hồ sơ hải quan trước thời điểm cơ quan hải quan thông báo kiểm tra thực tế hàng hóa hoặc kiểm tra chi tiết hồ sơ.",
        "status": "con_hieu_luc",
        "superseded_by": None
    },
    {
        "law_number": "08/2015/NĐ-CP",
        "article_number": "Điều 25",
        "title": "Khai hải quan điện tử qua hệ thống VNACCS/VCIS",
        "content": "Điều 25. Khai hải quan điện tử\n1. Khai hải quan điện tử được thực hiện thông qua Hệ thống xử lý dữ liệu điện tử hải quan (VNACCS/VCIS).\n2. Người khai hải quan lập tờ khai hải quan điện tử theo đúng tiêu chí, định dạng quy định và gửi đến Hệ thống.\n3. Hệ thống tự động tiếp nhận, kiểm tra, cấp số tờ khai và phân luồng (Luồng Xanh, Luồng Vàng, Luồng Đỏ) đối với tờ khai hải quan điện tử.",
        "status": "con_hieu_luc",
        "superseded_by": None
    },
    {
        "law_number": "128/2020/NĐ-CP",
        "article_number": "Điều 8",
        "title": "Xử phạt vi phạm quy định về khai hải quan (CŨ)",
        "content": "Điều 8. Xử phạt vi phạm quy định về khai hải quan\n1. Phạt tiền từ 1.000.000 đồng đến 2.000.000 đồng đối với hành vi khai hải quan không đúng thời hạn quy định.\n2. Phạt tiền từ 2.000.000 đồng đến 4.000.000 đồng đối với hành vi không khai bổ sung đúng thời hạn.\n[CẢNH BÁO PHÁP LÝ]: Văn bản 128/2020/NĐ-CP đã bị THAY THẾ bởi Nghị định 169/2026/NĐ-CP. Hãy tham chiếu quy định mới tại 169/2026/NĐ-CP.",
        "status": "bi_thay_the",
        "superseded_by": "169/2026/NĐ-CP"
    },
    {
        "law_number": "169/2026/NĐ-CP",
        "article_number": "Điều 10",
        "title": "Xử phạt vi phạm quy định về khai hải quan và thủ tục hải quan (MỚI 2026)",
        "content": "Điều 10. Xử phạt vi phạm quy định về khai hải quan và thủ tục hải quan điện tử\n1. Phạt tiền từ 2.000.000 đồng đến 5.000.000 đồng đối với hành vi chậm nộp/chậm khai báo tờ khai hải quan điện tử quá thời hạn 15 ngày.\n2. Phạt tiền từ 5.000.000 đồng đến 10.000.000 đồng đối với hành vi khai sai tên hàng, mã số HS, thuế suất dẫn đến thiếu số tiền thuế phải nộp.\n[QUY ĐỊNH MỚI 2026]: Nghị định 169/2026/NĐ-CP chính thức áp dụng thay thế cho Nghị định 128/2020/NĐ-CP.",
        "status": "con_hieu_luc",
        "superseded_by": None
    },
    {
        "law_number": "33/2023/TT-BTC",
        "article_number": "Điều 4",
        "title": "Khai và nộp chứng từ chứng nhận xuất xứ hàng hóa (C/O)",
        "content": "Điều 4. Kiểm tra, xác định xuất xứ hàng hóa xuất khẩu, nhập khẩu\n1. Người khai hải quan phải nộp chứng từ chứng nhận xuất xứ hàng hóa (Giấy chứng nhận xuất xứ C/O hoặc Chứng từ tự chứng nhận xuất xứ) cho cơ quan hải quan khi làm thủ tục hải quan.\n2. Cơ quan hải quan kiểm tra tính hợp lệ của C/O dựa trên đối chiếu thông tin trên tờ khai hải quan, chứng từ kèm theo và mã số HS.",
        "status": "con_hieu_luc",
        "superseded_by": None
    },
    {
        "law_number": "121/2025/TT-BTC",
        "article_number": "Điều 12",
        "title": "Quy trình kiểm tra sau thông quan đối với doanh nghiệp (MỚI 2025)",
        "content": "Điều 12. Kiểm tra sau thông quan tại trụ sở cơ quan hải quan và trụ sở doanh nghiệp\n1. Cơ quan hải quan thực hiện kiểm tra sau thông quan nhằm đánh giá tính chính xác, trung thực của hồ sơ hải quan đã thông quan trong thời hạn 05 năm kể từ ngày đăng ký tờ khai.\n2. Quyết định kiểm tra phải được gửi cho doanh nghiệp trước ít nhất 03 ngày làm việc.",
        "status": "con_hieu_luc",
        "superseded_by": None
    }
]

def extract_text_from_file(filepath: str) -> str:
    """
    Rút gọn và trích xuất văn bản từ các file .doc / .docx (bỏ qua file PDF).
    """
    if not filepath.endswith(('.doc', '.docx')):
        return ""

    # 1. Thử giải nén ZIP XML đối với định dạng .docx
    try:
        with zipfile.ZipFile(filepath, 'r') as docx:
            xml_content = docx.read('word/document.xml')
            root = ET.fromstring(xml_content)
            namespaces = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
            paragraphs = []
            for p in root.findall('.//w:p', namespaces):
                texts = [t.text for t in p.findall('.//w:t', namespaces) if t.text is not None]
                if texts:
                    paragraphs.append("".join(texts))
            if paragraphs:
                return "\n".join(paragraphs)
    except Exception:
        pass

    # 2. Thử đọc file .doc dạng HTML / Raw Binary Text
    try:
        with open(filepath, 'rb') as f:
            raw_bytes = f.read()

        text_str = raw_bytes.decode('utf-8', errors='ignore')

        # Nếu chứa HTML tags
        if '<html' in text_str.lower() or '<body' in text_str.lower() or '<table' in text_str.lower():
            clean_text = re.sub(r'<[^>]+>', ' ', text_str)
            clean_text = re.sub(r'&nbsp;', ' ', clean_text)
            clean_text = re.sub(r'\s+', ' ', clean_text)
            return clean_text

        # Decode chuỗi tiếng Việt từ Binary DOC
        clean_text = re.sub(r'[^\w\s\.\,\:\;\-\(\)\/\%\d\À-ỹ]', ' ', text_str)
        lines = [line.strip() for line in clean_text.splitlines() if len(line.strip()) > 15]
        return "\n".join(lines)
    except Exception as e:
        logger.warning(f"Không thể đọc file {filepath}: {str(e)}")
        return ""

def parse_articles_from_text(text: str, law_number: str) -> list[dict]:
    articles = []
    pattern = r'(Điều\s+\d+[\.\:]?\s*[^\n]*)'
    splits = re.split(pattern, text)
    
    if len(splits) <= 1:
        return articles

    for i in range(1, len(splits), 2):
        header = splits[i].strip()
        body = splits[i+1].strip() if i+1 < len(splits) else ""
        
        m_num = re.search(r'Điều\s+(\d+)', header)
        art_num = f"Điều {m_num.group(1)}" if m_num else header[:20]
        
        articles.append({
            "law_number": law_number,
            "article_number": art_num,
            "title": header,
            "content": f"{header}\n{body[:1500]}",
            "status": "con_hieu_luc",
            "superseded_by": None
        })
    return articles

async def seed_all_data():
    logger.info("=== BẮT ĐẦU CHẠY SEED DATA PIPELINE HẢI QUAN (DOCS ONLY) ===")
    await init_db()
    
    async with async_session() as session:
        logger.info("1. Đang nạp danh mục Văn bản Pháp luật (LegalDocument)...")
        for catalog in DOC_CATALOG:
            stmt = select(LegalDocument).where(LegalDocument.law_number == catalog["law_number"])
            res = await session.execute(stmt)
            existing_doc = res.scalars().first()
            if not existing_doc:
                doc_obj = LegalDocument(**catalog)
                session.add(doc_obj)
        await session.commit()

        raw_dir = os.path.join("data", "raw")
        if os.path.exists(raw_dir):
            for file_name in os.listdir(raw_dir):
                # Bỏ qua PDF theo yêu cầu doanh nghiệp (chỉ tập trung file Word .doc/.docx)
                if file_name.endswith((".doc", ".docx")):
                    file_path = os.path.join(raw_dir, file_name)
                    logger.info(f"Đang phân tích file Word: {file_name}")
                    text = extract_text_from_file(file_path)
                    if text:
                        law_num = "31/2022/TT-BTC" if "31_2022" in file_name else (
                            "08/2015/NĐ-CP" if "08_2015" in file_name else (
                                "01/2015/NĐ-CP" if "01_2015" in file_name else (
                                    "128/2020/NĐ-CP" if "128_2020" in file_name else (
                                        "12/2015/TT-BTC" if "12_2015" in file_name else (
                                            "33/2023/TT-BTC" if "33_2023" in file_name else (
                                                "68/2016/NĐ-CP" if "68_2016" in file_name else "Văn bản Hải quan"
                                            )
                                        )
                                    )
                                )
                            )
                        )
                        parsed = parse_articles_from_text(text, law_num)
                        for art in parsed[:5]:
                            PROCESSED_ARTICLES.append(art)

        logger.info("2. Đang lưu Điều/Khoản & tính toán Vector Embedding...")
        seeded_count = 0
        for art in PROCESSED_ARTICLES:
            stmt = select(CustomsDocument).where(
                CustomsDocument.law_number == art["law_number"],
                CustomsDocument.article_number == art["article_number"]
            )
            res = await session.execute(stmt)
            if not res.scalars().first():
                await add_document(
                    session=session,
                    content=art["content"],
                    law_number=art["law_number"],
                    article_number=art["article_number"],
                    title=art["title"],
                    status=art["status"],
                    superseded_by=art["superseded_by"]
                )
                
                legal_art = LegalArticle(
                    law_number=art["law_number"],
                    article_number=art["article_number"],
                    title=art["title"],
                    content=art["content"],
                    status=art["status"],
                    superseded_by=art["superseded_by"]
                )
                session.add(legal_art)
                seeded_count += 1

        await session.commit()
        logger.info(f"=== SEED DATA HOÀN TẤT: Đã nạp {seeded_count} Điều khoản vào SQL & pgvector DB ===")

if __name__ == "__main__":
    asyncio.run(seed_all_data())
