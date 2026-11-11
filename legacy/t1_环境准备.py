# 导入必要的库
import os
from pathlib import Path
from typing import List, Dict, Any

# 检查环境
print("检查环境...")
print(f"Python版本: {os.sys.version}")

# 模拟LlamaIndex和ChromaDB（如果未安装）
try:
    import llama_index
    import chromadb
    print("✅ LlamaIndex和ChromaDB已安装")
except ImportError:
    print("⚠️  部分库未安装，将使用模拟数据演示")
    print("   请运行: pip install llama-index-core chromadb")

# 创建示例数据目录
DATA_DIR = Path("./data/sample_documents")
DATA_DIR.mkdir(parents=True, exist_ok=True)
print(f"\n数据目录: {DATA_DIR.absolute()}")