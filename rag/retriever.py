from pathlib import Path
import argparse

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


# =========================
# 配置区
# =========================

VECTOR_DB_DIR = Path("/root/autodl-fs/qwen_api/vectory_db")

EMBEDDING_MODEL = "BAAI/bge-m3"

COLLECTION_NAME = "elden_ring_knowledge"


# =========================
# 加载向量库
# =========================

def load_vectorstore():
    if not VECTOR_DB_DIR.exists():
        raise FileNotFoundError(f"找不到向量库目录：{VECTOR_DB_DIR}")

    print(f"[INFO] 加载 Embedding 模型：{EMBEDDING_MODEL}")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={
            "device": "cpu"
        },
        encode_kwargs={
            "normalize_embeddings": True
        }
    )

    print(f"[INFO] 加载 Chroma 向量库：{VECTOR_DB_DIR}")

    vectorstore = Chroma(
        persist_directory=str(VECTOR_DB_DIR),
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )

    return vectorstore


# =========================
# 检索函数
# =========================

def retrieve(query: str, top_k: int = 3):
    vectorstore = load_vectorstore()

    print(f"\n[QUERY] {query}")
    print(f"[INFO] top_k = {top_k}")

    results = vectorstore.similarity_search_with_score(
        query=query,
        k=top_k
    )

    if not results:
        print("[WARN] 没有检索到相关内容")
        return []

    print("\n========== 检索结果 ==========\n")

    for idx, (doc, score) in enumerate(results, start=1):
        print(f"【结果 {idx}】")
        print(f"相似度距离 score: {score}")
        print(f"来源: {doc.metadata.get('source')}")
        print(f"文件: {doc.metadata.get('filename')}")
        print(f"分类: {doc.metadata.get('category')}")
        print(f"chunk_id: {doc.metadata.get('chunk_id')}")
        print("\n内容:")
        print(doc.page_content)
        print("\n" + "-" * 60 + "\n")

    return results


# =========================
# 命令行入口
# =========================

def main():
    parser = argparse.ArgumentParser(description="艾尔登法环 RAG 检索测试脚本")
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="用户问题"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
        help="返回的相关文本数量"
    )

    args = parser.parse_args()

    retrieve(args.query, args.top_k)


if __name__ == "__main__":
    main()
