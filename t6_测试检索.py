# 测试检索
from t5_模拟嵌入和检索 import SimpleVectorStore   # ① 导入类
from t3_文档加载器 import get_documents            # ① 导入文档加载函数

# ② 建库 + 装数据，只做一次（移出循环！）
vector_store = SimpleVectorStore()
documents = get_documents()
vector_store.add_documents(documents)              # 关键：不调用这行，检索永远是空

test_queries = [
    "Python是什么？",
    "RAG技术的优势",
    "向量数据库"
]

for query in test_queries:
    print(f"\n{'=' * 60}")
    print(f"查询: {query}")
    print(f"{'=' * 60}")

    results = vector_store.search(query, top_k=2)   # ③ 直接查，不再 new

    if results:
        for i, result in enumerate(results, 1):
            print(f"\n结果 {i} (相关度: {result['score']:.3f}):")
            print(f"{result['document'][:100]}...")
    else:
        print("未找到相关文档")