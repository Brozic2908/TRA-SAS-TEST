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
    docx_path = os.path.join("data", "raw", "31_2022_TT-BTC_343978.doc")
    text = get_docx_text(docx_path)
    
    print("=== TRÍCH XUẤT 3000 KÝ TỰ ĐẦU TIÊN ===")
    print(text[:3000])
