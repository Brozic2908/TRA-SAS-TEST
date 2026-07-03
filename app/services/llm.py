import json
import urllib.request
import urllib.error
import asyncio
import logging
from typing import Any, List, Optional
from pydantic import BaseModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from app.core.config import settings

logger = logging.getLogger("customs_rag")

# Embeddings đối tượng (Fallback nếu Google API key rỗng thì trả về 0.0 vector)
class FallbackEmbeddings:
    def __init__(self, google_key: str):
        self.google_key = google_key
        if google_key:
            self._gemini = GoogleGenerativeAIEmbeddings(
                model="models/gemini-embedding-001",
                google_api_key=google_key
            )
        else:
            self._gemini = None

    async def aembed_query(self, text: str) -> List[float]:
        if self._gemini:
            try:
                return await self._gemini.aembed_query(text)
            except Exception as e:
                logger.warning(f"Lỗi Gemini Embedding online ({str(e)}), trả về vector rỗng.")
        return [0.0] * 3072

embeddings = FallbackEmbeddings(settings.GOOGLE_API_KEY)

class GroqChatLLM(Runnable[Any, AIMessage]):
    """
    Groq LLM Client sử dụng mô hình Llama-3.3-70b-versatile (Miễn phí, siêu nhanh 300+ tokens/s).
    """
    def __init__(self, model: str = "llama-3.3-70b-versatile", temperature: float = 0.2, api_key: str = ""):
        super().__init__()
        self.model = model
        self.temperature = temperature
        self.api_key = api_key or settings.GROQ_API_KEY

    def with_structured_output(self, schema_cls: type) -> "GroqStructuredAdapter":
        return GroqStructuredAdapter(self, schema_cls)

    def invoke(self, input_data: Any, config: Optional[RunnableConfig] = None) -> AIMessage:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import nest_asyncio
                nest_asyncio.apply()
                return loop.run_until_complete(self.ainvoke(input_data, config))
            return loop.run_until_complete(self.ainvoke(input_data, config))
        except Exception:
            return asyncio.run(self.ainvoke(input_data, config))

    async def ainvoke(self, input_data: Any, config: Optional[RunnableConfig] = None, **kwargs) -> AIMessage:
        if not self.api_key:
            raise ValueError("Thiếu GROQ_API_KEY")

        messages = []
        if hasattr(input_data, "to_messages"):
            msgs = input_data.to_messages()
        elif isinstance(input_data, list):
            msgs = input_data
        else:
            msgs = [{"role": "user", "content": str(input_data)}]

        for m in msgs:
            role = getattr(m, "type", "user")
            if role in ["human", "user"]:
                role = "user"
            elif role == "system":
                role = "system"
            elif role in ["ai", "assistant"]:
                role = "assistant"
            
            content = getattr(m, "content", str(m))
            messages.append({"role": role, "content": content})

        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": 1500
        }

        models_to_try = [self.model, "llama-3.1-8b-instant", "mixtral-8x7b-32768"]
        # Loại bỏ trùng lặp nếu self.model đã là llama-3.1-8b-instant
        seen = set()
        unique_models = []
        for m in models_to_try:
            if m not in seen:
                seen.add(m)
                unique_models.append(m)

        def _sync_groq():
            last_error = None
            for model_name in unique_models:
                p_copy = dict(payload)
                p_copy["model"] = model_name
                req = urllib.request.Request(
                    url,
                    data=json.dumps(p_copy).encode("utf-8"),
                    headers=headers,
                    method="POST"
                )
                try:
                    with urllib.request.urlopen(req, timeout=12) as resp:
                        res_body = resp.read().decode("utf-8")
                        res_json = json.loads(res_body)
                        content = res_json["choices"][0]["message"]["content"]
                        if model_name != self.model:
                            logger.info(f"Đã fallback sang model Groq '{model_name}' thành công.")
                        return AIMessage(content=content)
                except urllib.error.HTTPError as err:
                    err_body = err.read().decode("utf-8", errors="ignore")
                    logger.warning(f"Lỗi Groq API model {model_name} HTTP {err.code}: {err_body}")
                    last_error = RuntimeError(f"Groq API Error {err.code}: {err_body}")
                    # Nếu lỗi 429 (rate limit) hoặc 503, thử model tiếp theo trong danh sách
                    if err.code in [429, 503, 404]:
                        continue
                    raise last_error
                except Exception as ex:
                    logger.warning(f"Lỗi kết nối Groq API với model {model_name}: {str(ex)}")
                    last_error = ex
                    continue

            if last_error:
                raise last_error
            raise RuntimeError("Không thể kết nối đến Groq API.")

        return await asyncio.to_thread(_sync_groq)

class GroqStructuredAdapter(Runnable[Any, Any]):
    """
    Adapter giúp Groq trả về JSON Structured Output tương thích với Pydantic.
    """
    def __init__(self, llm: Any, schema_cls: type):
        super().__init__()
        self.llm = llm
        self.schema_cls = schema_cls

    def invoke(self, input_data: Any, config: Optional[RunnableConfig] = None) -> Any:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import nest_asyncio
                nest_asyncio.apply()
                return loop.run_until_complete(self.ainvoke(input_data, config))
            return loop.run_until_complete(self.ainvoke(input_data, config))
        except Exception:
            return asyncio.run(self.ainvoke(input_data, config))

    async def ainvoke(self, input_data: Any, config: Optional[RunnableConfig] = None, **kwargs) -> Any:
        # Prompt ép LLM trả về đúng định dạng JSON Pydantic
        schema_json = json.dumps(self.schema_cls.model_json_schema(), ensure_ascii=False)
        sys_instruction = f"BẮT BUỘC trả về duy nhất 1 chuỗi JSON hợp lệ tuân thủ theo JSON Schema sau:\n{schema_json}\nKhông kèm theo lời giải thích hay markdown codeblock."

        if hasattr(input_data, "to_messages"):
            msgs = input_data.to_messages()
            msgs.insert(0, ("system", sys_instruction))
        else:
            input_data = f"{sys_instruction}\n\n{input_data}"

        res = await self.llm.ainvoke(input_data)
        text_content = res.content.strip()
        
        # Clean markdown wrappers ```json ... ```
        if text_content.startswith("```"):
            lines = text_content.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            text_content = "\n".join(lines).strip()

        data_dict = json.loads(text_content)
        return self.schema_cls(**data_dict)

class OpenRouterChatLLM(Runnable[Any, AIMessage]):
    """
    OpenRouter LLM Client hỗ trợ DeepSeek R1, Qwen3, v.v.
    URL: https://openrouter.ai - Miễn phí cho nhiều model mạnh.
    """
    def __init__(
        self,
        model: str = "deepseek/deepseek-r1:free",
        temperature: float = 0.2,
        api_key: str = ""
    ):
        super().__init__()
        self.model = model
        self.temperature = temperature
        self.api_key = api_key or settings.OPENROUTER_API_KEY

    def with_structured_output(self, schema_cls: type) -> "GroqStructuredAdapter":
        # Tái dụng GroqStructuredAdapter vì cùng giao thức OpenAI
        return GroqStructuredAdapter(self, schema_cls)

    def invoke(self, input_data: Any, config: Optional[RunnableConfig] = None) -> AIMessage:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import nest_asyncio
                nest_asyncio.apply()
                return loop.run_until_complete(self.ainvoke(input_data, config))
            return loop.run_until_complete(self.ainvoke(input_data, config))
        except Exception:
            return asyncio.run(self.ainvoke(input_data, config))

    async def ainvoke(self, input_data: Any, config: Optional[RunnableConfig] = None, **kwargs) -> AIMessage:
        if not self.api_key:
            raise ValueError("Thiếu OPENROUTER_API_KEY")

        messages = []
        if hasattr(input_data, "to_messages"):
            msgs = input_data.to_messages()
        elif isinstance(input_data, list):
            msgs = input_data
        else:
            msgs = [{"role": "user", "content": str(input_data)}]

        for m in msgs:
            role = getattr(m, "type", "user")
            if role in ["human", "user"]:
                role = "user"
            elif role == "system":
                role = "system"
            elif role in ["ai", "assistant"]:
                role = "assistant"
            content = getattr(m, "content", str(m))
            messages.append({"role": role, "content": content})

        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/Brozic2908/TRA-SAS-TEST",
            "X-Title": "TRA-SAS Customs RAG"
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": 2000
        }

        def _sync_openrouter():
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=20) as resp:
                res_body = resp.read().decode("utf-8")
                res_json = json.loads(res_body)
                # DeepSeek R1 có thể có reasoning tokens riêng, lấy content chính
                choice = res_json["choices"][0]["message"]
                content = choice.get("content") or ""
                # Bỏ qua phần <think>...</think> nếu có
                if "<think>" in content and "</think>" in content:
                    content = content.split("</think>")[-1].strip()
                return AIMessage(content=content)

        return await asyncio.to_thread(_sync_openrouter)


def get_grader_llm():
    """
    Model Routing:
    1️⃣ OpenRouter DeepSeek R1 (nếu có OPENROUTER_API_KEY)
    2️⃣ Groq Llama 3.3 70B (nếu có GROQ_API_KEY)
    3️⃣ Gemini Flash 2.0 (nếu có GOOGLE_API_KEY)
    """
    if settings.OPENROUTER_API_KEY:
        logger.info("[LLM Router] Sử dụng OpenRouter DeepSeek R1 (Grader)")
        # Dùng model nhẹ hơn cho grader để tiết kiệm quota
        return OpenRouterChatLLM(model="deepseek/deepseek-r1:free", temperature=0.0)
    if settings.GROQ_API_KEY:
        logger.info("[LLM Router] Sử dụng Groq Llama 3.3 70B (Grader)")
        return GroqChatLLM(model="llama-3.3-70b-versatile", temperature=0.0)
    logger.info("[LLM Router] Sử dụng Gemini Flash 2.0 (Grader)")
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.0,
        google_api_key=settings.GOOGLE_API_KEY,
        max_retries=1
    )

def get_generator_llm():
    """
    Model Routing:
    1️⃣ OpenRouter DeepSeek R1 (nếu có OPENROUTER_API_KEY)
    2️⃣ Groq Llama 3.3 70B (nếu có GROQ_API_KEY)
    3️⃣ Gemini Flash 2.0 (nếu có GOOGLE_API_KEY)
    """
    if settings.OPENROUTER_API_KEY:
        logger.info("[LLM Router] Sử dụng OpenRouter DeepSeek R1 (Generator)")
        return OpenRouterChatLLM(model="deepseek/deepseek-r1:free", temperature=0.3)
    if settings.GROQ_API_KEY:
        logger.info("[LLM Router] Sử dụng Groq Llama 3.3 70B (Generator)")
        return GroqChatLLM(model="llama-3.3-70b-versatile", temperature=0.2)
    logger.info("[LLM Router] Sử dụng Gemini Flash 2.0 (Generator)")
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.3,
        google_api_key=settings.GOOGLE_API_KEY,
        max_retries=1
    )
