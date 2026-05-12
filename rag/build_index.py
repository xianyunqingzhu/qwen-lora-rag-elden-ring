
from pathlib import Path
import shutil

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


# =========================
# 配置区
# =========================

KNOWLEDGE_DIR = Path("/root/autodl-fs/qwen_api/knowledge/lore")

VECTOR_DB_DIR = Path("/root/autodl-fs/qwen_api/vectory_db")

EMBEDDING_MODEL = "BAAI/bge-m3"

COLLECTION_NAME = "elden_ring_knowledge"


# =========================
# 获取 category
# =========================

def get_category(md_path: Path) -> str:
    """
    根据相对路径生成分类。
    例如：
    lore/人物志/女武神玛莲妮娜.md -> 人物志
    lore/背景剧情.md -> lore
    """
    rel_path = md_path.relative_to(KNOWLEDGE_DIR)

    if len(rel_path.parts) >= 2:
        return rel_path.parts[0]

    return "lore"


# =========================
# 加载所有 Markdown
# =========================

def load_markdown_documents():
    if not KNOWLEDGE_DIR.exists():
        raise FileNotFoundError(f"找不到知识库目录：{KNOWLEDGE_DIR}")

    md_files = sorted(KNOWLEDGE_DIR.rglob("*.md"))
    md_files = [
        p for p in md_files
        if ".ipynb_checkpoints" not in str(p)
    ]
    
    if not md_files:
        raise FileNotFoundError(f"没有找到 Markdown 文件：{KNOWLEDGE_DIR}")

    print(f"[INFO] 找到 Markdown 文件数量：{len(md_files)}")

    all_documents = []

    for md_path in md_files:
        print(f"[INFO] 读取：{md_path}")

        loader = TextLoader(
            str(md_path),
            encoding="utf-8"
        )

        docs = loader.load()

        category = get_category(md_path)

        for doc in docs:
            doc.metadata["source"] = str(md_path)
            doc.metadata["filename"] = md_path.name
            doc.metadata["stem"] = md_path.stem
            doc.metadata["category"] = category
            doc.metadata["relative_path"] = str(md_path.relative_to(KNOWLEDGE_DIR))

        all_documents.extend(docs)

    return all_documents


# =========================
# 构建索引
# =========================

def build_index():
    print(f"[INFO] 知识库目录：{KNOWLEDGE_DIR}")

    documents = load_markdown_documents()

    print(f"[INFO] 原始文档数量：{len(documents)}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
        separators=[
            "\n# ",
            "\n## ",
            "\n### ",
            "\n\n",
            "\n",
            "。",
            "！",
            "？",
            "；",
            "，",
            " ",
            ""
        ]
    )

    chunks = text_splitter.split_documents(documents)

    print(f"[INFO] 切分后 chunk 数量：{len(chunks)}")

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i

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

    if VECTOR_DB_DIR.exists():
        print(f"[INFO] 清空旧向量库：{VECTOR_DB_DIR}")
        shutil.rmtree(VECTOR_DB_DIR)

    VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] 写入 Chroma 向量库：{VECTOR_DB_DIR}")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(VECTOR_DB_DIR),
        collection_name=COLLECTION_NAME
    )

    vectorstore.persist()

    print("[SUCCESS] RAG 向量索引构建完成")
    print(f"[SUCCESS] Markdown 文件数：{len(documents)}")
    print(f"[SUCCESS] Chunk 数：{len(chunks)}")
    print(f"[SUCCESS] 向量库路径：{VECTOR_DB_DIR}")
    print(f"[SUCCESS] Collection：{COLLECTION_NAME}")


if __name__ == "__main__":
    build_index()
