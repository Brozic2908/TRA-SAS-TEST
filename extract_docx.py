import zipfile
import xml.etree.ElementTree as ET
import os

def get_docx_text(path):
    try:
        with zipfile.ZipFile(path) as docx:
            xml_content = docx.read('word/document.xml')
            root = ET.fromstring(xml_content)
            namespaces = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
            paragraphs = []
            for p in root.findall('.//w:p', namespaces):
                texts = []
                for t in p.findall('.//w:t', namespaces):
                    if t.text:
                        texts.append(t.text)
                if texts:
                    paragraphs.append("".join(texts))
            return "\n".join(paragraphs)
    except Exception as e:
        return f"Lỗi đọc file: {str(e)}"

if __name__ == "__main__":
    docx_path = os.path.join("data", "raw", "95_VBHN-VPQH_m_673593.docx")
    text = get_docx_text(docx_path)
    
    # In ra 3000 ký tự đầu tiên để xem cấu trúc và nội dung Điều khoản
    print("=== 3000 KÝ TỰ ĐẦU TIÊN CỦA LUẬT HẢI QUAN (95/VBHN-VPQH) ===")
    print(text[:3000])
    
    # Tìm các Điều liên quan đến thủ tục hải quan, khai hải quan, hoặc thông quan để trích xuất
    print("\n=== TRÍCH XUẤT CÁC ĐIỀU QUAN TRỌNG ===")
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if line.startswith("Điều 3.") or line.startswith("Điều 29.") or line.startswith("Điều 16."):
            print(f"\n--- {line} ---")
            # In ra 5 dòng tiếp theo của Điều đó
            for j in range(1, 8):
                if i + j < len(lines):
                    print(lines[i+j])
