from typing import List
import re
# class TextSplitter:
#     """
#     文本分块器
#     """
#
#     def __init__(self, chunk_size: int = 100, chunk_overlap: int = 20):
#         self.chunk_size = chunk_size
#         self.chunk_overlap = chunk_overlap
#
#     def split(self, text: str) -> List[str]:
#         """
#         将文本切分成块
#
#         Args:
#             text: 输入文本
#
#         Returns:
#             文本块列表
#         """
#         chunks = []
#         start = 0
#
#         while start < len(text):
#             end = start + self.chunk_size
#             chunk = text[start:end]
#
#             if chunk:
#                 chunks.append(chunk.strip())
#
#             # 移动窗口，保留overlap
#             start = end - self.chunk_overlap
#
#         return chunks


class TextSplitter:
    """
    文本分块器（按句子边界切分）
    """

    # 句子切分规则：匹配到句末标点(。！？!?)或换行为止
    SENTENCE_PATTERN = re.compile(r"[^。！？!?\n]*[。！？!?\n]|[^。！？!?\n]+$")

    def __init__(self, chunk_size: int = 100, chunk_overlap: int = 20):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str) -> List[str]:
        """
        按句子边界切分文本，尽量让每个块包含完整的句子
        """
        # ① 先把文本打成"句子"列表
        sentences = [s for s in self.SENTENCE_PATTERN.findall(text) if s.strip()]
        if not sentences:
            return []

        # ② 贪心打包：往当前块里塞句子，塞不下就收尾
        chunks = []
        buf = []
        size = 0

        for sent in sentences:
            # 估算加入这句后的长度
            new_size = len(sent) if not buf else size + len(sent)

            # 超限 且 当前块非空 → 收尾，开新块
            if new_size > self.chunk_size and buf:
                chunks.append("".join(buf))
                # 保留末尾若干句子作为重叠
                while buf and size > self.chunk_overlap:
                    size -= len(buf.pop(0))

            buf.append(sent)
            size += len(sent)

        # ③ 收尾：最后一块
        if buf:
            chunks.append("".join(buf))

        return chunks
# 测试分块器
if __name__ == '__main__':
    from t3_文档加载器 import get_documents
    splitter = TextSplitter(chunk_size=150, chunk_overlap=20)

    print("测试文本分块:\n")
    documents=get_documents()
    test_text = documents[2]['content']
    chunks = splitter.split(test_text)

    print(f"原文长度: {len(test_text)} 字符")
    print(f"分块数量: {len(chunks)} 块\n")

    for i, chunk in enumerate(chunks, 1):
        print(f"块 {i}: {chunk}\n")
