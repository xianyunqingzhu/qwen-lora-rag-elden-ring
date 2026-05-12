from pathlib import Path
import argparse
import requests

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


# =========================
# 配置区
# =========================

VECTOR_DB_DIR = Path("/root/autodl-fs/qwen_api/vectory_db")
EMBEDDING_MODEL = "BAAI/bge-m3"
COLLECTION_NAME = "elden_ring_knowledge"

VLLM_API_URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL_NAME = "/root/autodl-fs/models/qwen2.5-7b-elden-ring-merged"


# =========================
# 加载向量库
# =========================

def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    vectorstore = Chroma(
        persist_directory=str(VECTOR_DB_DIR),
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )

    return vectorstore


# =========================
# 检索知识
# =========================

def retrieve_context(query: str, top_k: int = 3) -> str:
    vectorstore = load_vectorstore()

    results = vectorstore.similarity_search_with_score(
        query=query,
        k=top_k
    )

    if not results:
        return ""

    context_parts = []

    for idx, (doc, score) in enumerate(results, start=1):
        source = doc.metadata.get("source", "未知来源")
        chunk_id = doc.metadata.get("chunk_id", "未知chunk")

        context_parts.append(
            f"【资料{idx}】\n"
            f"来源：{source}\n"
            f"chunk_id：{chunk_id}\n"
            f"内容：\n{doc.page_content}"
        )

    return "\n\n".join(context_parts)


# =========================
# 构造 Prompt
# =========================

def build_prompt(question: str, context: str) -> str:
    return f"""你是一个专业的《艾尔登法环》攻略与世界观助手。

请严格根据【参考资料】回答用户问题。

要求：
1. 如果参考资料中有答案，必须优先依据参考资料回答。
2. 如果参考资料不足，请明确说明“当前资料不足”，不要编造。
3. 不要把其他游戏的内容混入《艾尔登法环》。
4. 回答要结构化、清晰、适合新手理解。

【参考资料】
{context}

【用户问题】
{question}

请按以下格式回答：
一、简要结论
二、详细说明
三、补充说明
"""


# =========================
# 调用 vLLM
# =========================

def call_vllm(prompt: str, temperature: float = 0.3, max_tokens: int = 1024) -> str:
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    response = requests.post(
        VLLM_API_URL,
        json=payload,
        timeout=120
    )

    response.raise_for_status()

    data = response.json()

    return data["choices"][0]["message"]["content"]


# =========================
# RAG 问答
# =========================

def rag_chat(question: str, top_k: int = 3):
    print("\n[1] 正在检索知识库...")
    context = retrieve_context(question, top_k=top_k)

    if not context:
        print("[WARN] 未检索到相关资料")

    print("\n[2] 检索到的参考资料：")
    print("=" * 80)
    print(context)
    print("=" * 80)

    print("\n[3] 正在调用 vLLM 生成回答...")
    prompt = build_prompt(question, context)
    answer = call_vllm(prompt)

    print("\n[4] 模型回答：")
    print("=" * 80)
    print(answer)
    print("=" * 80)


# =========================
# 命令行入口
# =========================

def main():
    parser = argparse.ArgumentParser(description="艾尔登法环 RAG 问答脚本")
    parser.add_argument("--query", type=str, required=True, help="用户问题")
    parser.add_argument("--top_k", type=int, default=3, help="检索文档数量")

    args = parser.parse_args()

    rag_chat(args.query, top_k=args.top_k)


if __name__ == "__main__":
    main()
