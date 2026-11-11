import os
from openai import OpenAI

_api_key=os.environ.get('DEEPSEEK_API_KEY')
if not _api_key:
    raise RuntimeError("未找到 DEEPSEEK_API_KEY 环境变量，请先设置")

client = OpenAI(
    api_key=_api_key,
    base_url="https://api.deepseek.com",
)
from typing import Dict, List, Any
from vector_store import SimpleVectorStore
from doc_loader import get_documents
from text_splitter import TextSplitter

def _fmt_source(meta: Dict[str, Any]) -> str:
    label = meta.get('filename', '?')
    if 'page' in meta:
        label += f" 第{meta['page']}页"
    if 'slide' in meta:
        label += f" 第{meta['slide']}张"
    if 'chunk_index' in meta:
        label += f" 第{meta['chunk_index']}块"
    return label

class SimpleRAG:
    """
    简单的RAG系统
    """

    def __init__(self, documents: List[Dict[str, Any]], chunk_size: int = 350, chunk_overlap: int = 50):
        self.vector_store = SimpleVectorStore(reset=True)
        self.splitter = TextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        # 把每篇文档切成多个块，每个块携带自己的 metadata
        chunks = []
        for doc in documents:
            pieces = self.splitter.split(doc['content'])
            for i, piece in enumerate(pieces):
                chunks.append({
                    'content': piece,
                    'metadata': {
                        **doc['metadata'],  # 继承原文档的 metadata
                        'chunk_index': i,  # 记下这是第几块
                    }
                })

        print(f"分块完成: {len(documents)} 篇文档 -> {len(chunks)} 个块")
        self.vector_store.add_documents(chunks)

        # 模拟LLM响应
        self.llm_prompt_template = """
基于以下文档回答问题：

{context}

问题: {question}

答案:
        """

    def query(self, question: str, top_k: int =3) -> Dict:
        """
        RAG查询

        Args:
            question: 用户问题
            top_k: 检索文档数量

        Returns:
            包含答案和来源的字典
        """
        # 步骤1：检索
        print(f"\n🔍 步骤1: 检索相关文档")
        retrieved_docs = self.vector_store.search(question, top_k=top_k)

        if not retrieved_docs:
            return {
                'question': question,
                'answer': '抱歉，知识库中没有找到相关信息。',
                'sources': []
            }

        print(f"找到 {len(retrieved_docs)} 个相关文档")

        # 步骤2：构建上下文
        print(f"\n📝 步骤2: 构建上下文")
        context = "\n\n".join([
            f"[{i + 1}] 来源：{_fmt_source(doc['metadata'])}\n{doc['document']}"
            for i, doc in enumerate(retrieved_docs)
        ])
        # 步骤3：生成答案
        print(f"\n🤖 步骤3: 生成答案")
        answer = self._generate_answer(question,context)

        return {
            'question': question,
            'answer': answer,
            'sources': [
                {
                    'file': doc['metadata']['filename'],
                    'page': doc['metadata'].get('page', '-'),
                    'slide': doc['metadata'].get('slide', '-'),
                    'chunk_index': doc['metadata'].get('chunk_index', '-'),   # ← 块序号
                    'preview': doc['document'][:40],                          # ← 内容预览
                    'score': doc['score']
                }
                for doc in retrieved_docs
            ]
        }

    # def _generate_answer(self, question: str, context: str) -> str:
    #     """
    #     生成答案（简化版）
    #     """
    #     # 简化实现：基于规则生成答案
    #     if "Python" in question and "Python" in context:
    #         return "根据文档，Python是一种高级编程语言，由Guido van Rossum于1991年创建，广泛应用于Web开发、数据科学和人工智能。"
    #     elif "RAG" in question:
    #         return "RAG（Retrieval-Augmented Generation）结合了信息检索和生成模型，可以减少LLM的幻觉问题，提高答案准确性。"
    #     else:
    #         return f"根据检索到的信息，{context[:100]}..."

    # def _generate_answer(self,question,context):
    #     # 干掉所有 if-elif，让检索结果赤裸裸地暴露出来
    #     return f"[未接入LLM，仅回显检索片段]\n{context}"

    def _generate_answer(self, question, context):
            # 系统提示词：约束模型只能用给定资料，不许自由发挥
            system_prompt = """你是一个严谨的文档问答助手。

      请严格遵守以下规则：
      1. 只依据【参考资料】中的内容回答问题，不要使用你自己的知识。
      2. 如果参考资料中没有相关信息，直接回答"根据已知资料无法回答该问题"，不要编造。
      3. 回答要简洁准确，不要输出与问题无关的内容。
      4. 禁止拼接：如果参考资料分别描述了 A 和 B，但没有直接说明两者的关系或区别，
         不要自行拼出一个对比。此时应回答"资料中没有直接对比 A 和 B 的内容"。
      5. 每条要点都必须能在参考资料中找到对应原句；找不到依据的要点不要写。
      6. 回答前先判断：参考资料是否"直接支持"这个问题？
         如果只是"分别相关"而不是"直接回答"，就说明资料不足。"""

            # 用户消息：把资料和问题组织清楚，用标记分隔
            user_prompt = f"""【参考资料】
           {context}

           【问题】
           {question}"""

            # 调用大模型
            resp = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
            )

            return resp.choices[0].message.content

# 创建RAG系统
documents=get_documents()
rag_system = SimpleRAG(documents)
print("RAG系统初始化完成！")
