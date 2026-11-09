import os
import shutil
from typing import List, Dict, Any

import numpy as np
from openai import OpenAI
import chromadb

# ---- 嵌入模型客户端 ----
_api_key = os.environ.get("SILICONFLOW_API_KEY")
if not _api_key:
    raise RuntimeError("未找到 SILICONFLOW_API_KEY 环境变量，请先设置")

_embed_client = OpenAI(
    api_key=_api_key,
    base_url="https://api.siliconflow.cn/v1",
)

EMBED_MODEL = "BAAI/bge-m3"
EMBED_BATCH = 32

# ---- Chroma 持久化配置 ----
DB_DIR = "chroma_db"          # 磁盘上的存储目录
COLLECTION = "rag_demo"       # 集合名（相当于数据库里的"表"）


class EmbeddingModel:
    """把文本转成向量（调用嵌入 API）"""

    def __init__(self, model: str = EMBED_MODEL):
        self.model = model

    def embed(self, texts: List[str]) -> np.ndarray:
        """批量把文本转成向量，返回 (n, 1024) 的数组"""
        vectors = []
        for i in range(0, len(texts), EMBED_BATCH):
            batch = texts[i:i + EMBED_BATCH]
            resp = _embed_client.embeddings.create(
                model=self.model,
                input=batch,
            )
            vectors.extend([d.embedding for d in resp.data])
        return np.array(vectors, dtype=np.float32)

def _safe_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Chroma metadata 只接受 str/int/float/bool，且不能为空"""
    out = {}
    for k, v in meta.items():
        if isinstance(v, (str, int, float, bool)):
            out[k] = v
        elif v is None:
            continue
        else:
            out[k] = str(v)
    return out

def _normalize(mat: np.ndarray) -> np.ndarray:
    """按行做 L2 归一化，这样点积就等于余弦相似度"""
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    return mat / norms


class SimpleVectorStore:
    """
    向量存储（Chroma 持久化版）
    接口与之前完全一致：add_documents / search
    """

    def __init__(self, reset: bool = False):
        # reset=True 时清空磁盘上的旧数据，强制重新建库
        if reset and os.path.exists(DB_DIR):
            shutil.rmtree(DB_DIR)
            print(f"  [Chroma] 已清空旧存储 {DB_DIR}/")

        self.embedder = EmbeddingModel()
        self.client = chromadb.PersistentClient(path=DB_DIR)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION,
            metadata={"hnsw:space": "cosine"},   # 用余弦距离
        )
        self.documents = []      # 当前库里的文档（供查元数据用）

    def add_documents(self, docs: List[Dict[str, Any]]):
        """添加文档到向量存储（已存在则跳过嵌入）"""
        self.documents = docs

        # 库里有数据 → 直接复用，跳过嵌入
        if self.collection.count() > 0:
            print(f"  [Chroma] 复用已有索引，共 {self.collection.count()} 个块（跳过嵌入）")
            return

        texts = [doc['content'] for doc in docs]
        print(f"  [Chroma] 首次建库，正在向量化 {len(texts)} 个块...")
        vectors = _normalize(self.embedder.embed(texts))

        self.collection.add(
            ids=[f"chunk_{i}" for i in range(len(docs))],
            embeddings=[v.tolist() for v in vectors],
            documents=texts,
            metadatas=[_safe_meta(d['metadata']) for d in docs],
        )
        print(f"  [Chroma] 已存入磁盘：{DB_DIR}/，共 {self.collection.count()} 个块")

    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """搜索相关文档"""
        if self.collection.count() == 0:
            return []

        # 查询向量（同样要归一化）
        query_vector = _normalize(self.embedder.embed([query]))

        # 交给 Chroma 检索
        n = min(top_k, self.collection.count())
        res = self.collection.query(
            query_embeddings=[query_vector[0].tolist()],
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )

        results = []
        for text, meta, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            results.append({
                'document': text,
                'metadata': meta,
                # Chroma 返回的是"余弦距离"，相似度 = 1 - 距离
                'score': 1.0 - float(dist),
            })
        return results


if __name__ == '__main__':
    from t3_文档加载器 import get_documents
    from t4_文本分块器 import TextSplitter

    splitter = TextSplitter(chunk_size=350, chunk_overlap=50)
    documents = get_documents()

    chunks = []
    for doc in documents:
        for i, piece in enumerate(splitter.split(doc['content'])):
            chunks.append({
                'content': piece,
                'metadata': {**doc['metadata'], 'chunk_index': i},
            })

    vector_store = SimpleVectorStore(reset=True)
    vector_store.add_documents(chunks)

    print(f"\n向量库就绪，共 {len(chunks)} 个块")

    results = vector_store.search("什么是RAG？", top_k=2)
    for i, r in enumerate(results, 1):
        meta = r['metadata']
        loc = ""
        if 'page' in meta:
            loc += f" 第{meta['page']}页"
        if 'slide' in meta:
            loc += f" 第{meta['slide']}张"
        print(f"\n结果 {i} (相似度 {r['score']:.4f}): {meta['filename']}{loc}")
        print(f"  {r['document'][:60]}")