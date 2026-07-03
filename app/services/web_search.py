import json
import urllib.request
import asyncio
import logging
from app.core.config import settings

logger = logging.getLogger("customs_rag")

def _sync_tavily_search(query: str, api_key: str) -> list[str]:
    """
    Hàm gọi đồng bộ Tavily Search API.
    """
    if not api_key:
        logger.warning("TAVILY_API_KEY trống. Sử dụng dữ liệu giả lập cho Web Search.")
        return [f"Kết quả giả lập tìm kiếm web cho từ khóa: {query}"]
    
    url = "https://api.tavily.com/search"
    headers = {"Content-Type": "application/json"}
    payload = {
        "api_key": api_key,
        "query": query,
        "search_depth": "basic",
        "max_results": 3
    }
    
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST"
    )
    
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            res_body = response.read().decode("utf-8")
            res_json = json.loads(res_body)
            results = res_json.get("results", [])
            contents = [r.get("content", "") for r in results if r.get("content")]
            if not contents:
                return [f"Không tìm thấy kết quả tìm kiếm nào trên web cho: {query}"]
            return contents
    except Exception as e:
        logger.error(f"Lỗi kết nối Tavily API: {str(e)}")
        return [f"Lỗi tìm kiếm web cho từ khóa: {query} ({str(e)})"]

async def web_search(query: str) -> list[str]:
    """
    Tìm kiếm web bất đồng bộ bằng cách chạy luồng phụ (asyncio.to_thread).
    """
    return await asyncio.to_thread(_sync_tavily_search, query, settings.TAVILY_API_KEY)
