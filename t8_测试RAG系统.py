# 测试RAG系统
from t7_完整的RAG查询系统 import rag_system

test_questions = [
    "什么时候你真的需要RAG?"
]

for q in test_questions:
    result = rag_system.query(q)

    print(f"\n{'=' * 60}")
    print(f"问题: {result['question']}")
    print(f"\n答案: {result['answer']}")
    print(f"\n来源: {len(result['sources'])} 个文档")
    for source in result['sources']:
        loc = ""
        if source.get('page', '-') != '-':
            loc += f" 第{source['page']}页"
        if source.get('slide', '-') != '-':
            loc += f" 第{source['slide']}张"
        print(f"  - {source['file']}{loc} 块{source['chunk_index']} (相关度: {source['score']:.3f})")
        print(f"      {source['preview']}...")
    print()
    print("-" * 40)
