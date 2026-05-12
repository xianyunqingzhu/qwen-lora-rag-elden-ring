from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


app = FastAPI(title="Qwen Elden Ring RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# =========================
# vLLM 配置
# =========================

client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1"
)

MODEL_NAME = "/root/autodl-fs/models/qwen2.5-7b-elden-ring-merged"


# =========================
# RAG 配置
# =========================

VECTOR_DB_DIR = Path("/root/autodl-fs/qwen_api/vectory_db")
EMBEDDING_MODEL = "BAAI/bge-m3"
COLLECTION_NAME = "elden_ring_knowledge"


# =========================
# 请求 / 响应模型
# =========================

class ChatRequest(BaseModel):
    question: str
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 512


class RagChatRequest(BaseModel):
    question: str
    top_k: Optional[int] = 4
    temperature: Optional[float] = 0.3
    max_tokens: Optional[int] = 1024
    debug: Optional[bool] = False

# =========================
# 初始化 RAG
# =========================

print("[INFO] 正在加载 Embedding 模型...")

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

print("[INFO] 正在加载 Chroma 向量库...")

vectorstore = Chroma(
    persist_directory=str(VECTOR_DB_DIR),
    embedding_function=embeddings,
    collection_name=COLLECTION_NAME
)

print("[SUCCESS] RAG 初始化完成")


# =========================
# Prompt 构建
# =========================

def build_rag_prompt(question: str, context: str) -> str:
    return f"""你是一个专业的《艾尔登法环》攻略与世界观助手。

请严格根据【参考资料】回答用户问题。

要求：
1. 如果参考资料中有答案，必须优先依据参考资料回答。
2. 如果参考资料不足，请明确说明“当前资料不足”，不要编造。
3. 不要把其他游戏的内容混入《艾尔登法环》。
4. 回答要结构化、清晰、适合新手理解。
5. 如果涉及剧情、人物关系或设定，请尽量说明依据。

【参考资料】
{context}

【用户问题】
{question}

请按以下格式回答：
一、简要结论
二、详细说明
三、补充说明
"""


def retrieve_context(question: str, top_k: int):
    results = vectorstore.similarity_search_with_score(
        query=question,
        k=top_k
    )

    context_parts = []
    sources = []

    for idx, (doc, score) in enumerate(results, start=1):
        metadata = doc.metadata or {}

        source_item = {
            "rank": idx,
            "score": float(score),
            "source": metadata.get("source"),
            "filename": metadata.get("filename"),
            "category": metadata.get("category"),
            "chunk_id": metadata.get("chunk_id"),
            "content": doc.page_content
        }

        sources.append(source_item)

        context_parts.append(
            f"【资料{idx}】\n"
            f"来源：{source_item['source']}\n"
            f"文件：{source_item['filename']}\n"
            f"分类：{source_item['category']}\n"
            f"chunk_id：{source_item['chunk_id']}\n"
            f"内容：\n{doc.page_content}"
        )

    context = "\n\n".join(context_parts)

    return context, sources


# =========================
# 普通聊天接口
# =========================

@app.post("/chat")
def chat(req: ChatRequest):
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": "你是一个专业、简洁的中文AI助手。"
            },
            {
                "role": "user",
                "content": req.question
            }
        ],
        temperature=req.temperature,
        max_tokens=req.max_tokens
    )

    answer = response.choices[0].message.content

    return {
        "mode": "normal",
        "question": req.question,
        "answer": answer
    }


# =========================
# RAG 聊天接口
# =========================

@app.post("/rag/chat")
def rag_chat(req: RagChatRequest):
    context, sources = retrieve_context(
        question=req.question,
        top_k=req.top_k
    )

    prompt = build_rag_prompt(
        question=req.question,
        context=context
    )

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=req.temperature,
        max_tokens=req.max_tokens
    )

    answer = response.choices[0].message.content

    result = {
        "mode": "rag",
        "question": req.question,
        "answer": answer
    }
    
    if req.debug:
        result["top_k"] = req.top_k
        result["sources"] = sources
    
    return result


# =========================
# 健康检查
# =========================

@app.get("/")
def root():
    return {
        "message": "Qwen Elden Ring RAG API is running",
        "model": MODEL_NAME
    }