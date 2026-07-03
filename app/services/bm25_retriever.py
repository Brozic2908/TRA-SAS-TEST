import re
import math
from typing import List, Dict, Any

def tokenize_vietnamese(text: str) -> List[str]:
    """
    Tách từ đơn giản cho tiếng Việt (lowercase, bỏ dấu câu đặc biệt).
    """
    cleaned = re.sub(r'[^\w\s]', ' ', text.lower())
    tokens = [t for t in cleaned.split() if len(t) > 1]
    return tokens

class SimpleBM25:
    """
    Bộ tìm kiếm BM25 (Best Matching 25) thuần Python cho offline fallback.
    """
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_len = []
        self.avgdl = 0.0
        self.doc_tokens = []
        self.docs = []
        self.idf = {}

    def fit(self, docs: List[Dict[str, Any]]):
        """
        Huấn luyện index BM25 trên danh sách các tài liệu (mỗi dict có trường 'content').
        """
        self.docs = docs
        self.doc_tokens = []
        self.doc_len = []
        df = {}
        N = len(docs)
        if N == 0:
            return

        for doc in docs:
            tokens = tokenize_vietnamese(doc.get("content", ""))
            self.doc_tokens.append(tokens)
            self.doc_len.append(len(tokens))
            unique_tokens = set(tokens)
            for t in unique_tokens:
                df[t] = df.get(t, 0) + 1

        self.avgdl = sum(self.doc_len) / N if N > 0 else 1.0

        for t, freq in df.items():
            self.idf[t] = math.log((N - freq + 0.5) / (freq + 0.5) + 1)

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Tìm kiếm các tài liệu tương quan nhất với query.
        """
        if not self.docs:
            return []

        query_tokens = tokenize_vietnamese(query)
        scores = []

        for idx, doc in enumerate(self.docs):
            tokens = self.doc_tokens[idx]
            doc_l = self.doc_len[idx]
            score = 0.0
            
            # Tính tần suất từ trong tài liệu (TF)
            tf_map = {}
            for t in tokens:
                tf_map[t] = tf_map.get(t, 0) + 1

            for qt in query_tokens:
                if qt in tf_map:
                    tf = tf_map[qt]
                    idf = self.idf.get(qt, 0.1)
                    num = tf * (self.k1 + 1)
                    denom = tf + self.k1 * (1 - self.b + self.b * (doc_l / self.avgdl))
                    score += idf * (num / denom)

            scores.append((score, doc))

        scores.sort(key=lambda x: x[0], reverse=True)
        return [doc for score, doc in scores[:top_k] if score > 0]

# Single instance global store for offline BM25
bm25_index = SimpleBM25()
