# legacy

早期学习阶段的脚本，保留作参考，**不参与当前流程**。

| 文件 | 说明 |
|---|---|
| `t1_环境准备.py` | 环境自检。注意：其中 `import llama_index` 是**错误写法** —— 安装 `llama-index-core` 后的正确导入是 `from llama_index.core import ...` |
| `t2_创建示例文档.py` | 生成早期样例 `doc1~doc4.txt`。现语料已替换为真实课程 PDF |
| `t6_测试检索.py` | 早期检索测试，已被 `t8_测试RAG系统.py` 取代 |
| `test.py` | TF-IDF 原理练手，与项目无关 |