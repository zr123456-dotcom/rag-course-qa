from pathlib import Path
from typing import List, Dict, Any
import re

from pypdf import PdfReader
from docx import Document as DocxDocument
from pptx import Presentation as PptxPresentation
from pptx.enum.shapes import MSO_SHAPE_TYPE


SUPPORTED = {".txt", ".md", ".pdf", ".docx",".pptx"}

# 行首是这些 → 视为新的条目，不合并
_LIST_START = re.compile(r"^\s*(?:[0-9０-９]+\s*[、.．)）]|（[0-9０-９]+）|[-*•·]|[一二三四五六七八九十]+[、.])")
# 行尾是这些 → 句子已结束，不合并8
_SENT_END = ("。", "！", "？", "；", "：", ".", "!", "?", ";", ":", "）", ")", "】", "》")


def _join_soft_wraps(text: str) -> str:
    """把 PDF 的视觉换行合并回自然段落"""
    lines = [ln.rstrip() for ln in text.split("\n")]
    out: List[str] = []
    for ln in lines:
        if not ln.strip():
            out.append("")
            continue
        prev = out[-1] if out else ""
        merge = (
            prev != ""                                  # 上一行非空
            and not prev.endswith(_SENT_END)            # 上一行没结束句子
            and not _LIST_START.match(ln)               # 本行不是新条目
        )
        if merge:
            # 中文之间不加空格，中英文之间加一个
            need_space = (prev[-1].isascii() and ln[0].isascii())
            out[-1] = prev + (" " if need_space else "") + ln.strip()
        else:
            out.append(ln.strip())
    return "\n".join(out)


def _clean_text(text: str) -> str:
    """文本规范化：统一换行、压缩空白、去掉多余空行"""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\u3000]+", " ", text)      # 连续空格/全角空格 → 一个空格
    text = re.sub(r"\n{3,}", "\n\n", text)          # 3 个以上换行 → 2 个
    return text.strip()


class SimpleDocumentLoader:
    """
    多格式文档加载器
    支持: .txt / .md / .pdf / .docx
    """

    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.errors: List[Dict[str, str]] = []      # 记录失败的文件

    # ---------- 对外接口 ----------
    def load(self) -> List[Dict[str, Any]]:
        documents = []
        files = sorted(
            p for p in self.directory.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED
        )
        for path in files:
            try:
                docs = self._load_one(path)
                print(f"  [加载] {path.name} -> {len(docs)} 个文档单元")
                documents.extend(docs)
            except Exception as exc:
                # 单个文件失败，不影响其他文件
                msg = f"{type(exc).__name__}: {exc}"
                self.errors.append({"file": path.name, "error": msg})
                print(f"  [跳过] {path.name} -> {msg}")
        return documents

    # ---------- 格式分发 ----------
    def _load_one(self, path: Path) -> List[Dict[str, Any]]:
        suffix = path.suffix.lower()
        if suffix in (".txt", ".md"):
            return self._load_text(path)
        if suffix == ".pdf":
            return self._load_pdf(path)
        if suffix == ".docx":
            return self._load_docx(path)
        if suffix ==".pptx":
            return self._load_pptx(path)
        return []

    # ---------- 各格式实现 ----------
    def _load_text(self, path: Path) -> List[Dict[str, Any]]:
        raw = None
        for enc in ("utf-8", "gbk", "utf-16"):
            try:
                raw = path.read_text(encoding=enc)
                break
            except (UnicodeDecodeError, UnicodeError):
                continue
        if raw is None:
            raise ValueError("无法识别编码（试过 utf-8 / gbk / utf-16）")
        text = _clean_text(raw)
        if not text:
            raise ValueError("文件内容为空")
        return [self._make_doc(text, path, {})]

    def _load_pdf(self, path: Path) -> List[Dict[str, Any]]:
        reader = PdfReader(str(path))
        docs = []
        for page_no, page in enumerate(reader.pages, start=1):
            raw = page.extract_text() or ""
            text = _clean_text(_join_soft_wraps(raw))
            if not text:
                continue                      # 该页无文本层（可能是扫描页）
            docs.append(self._make_doc(text, path, {"page": page_no}))
        if not docs:
            raise ValueError("未提取到任何文本（可能是扫描件，需要 OCR）")
        return docs

    def _load_docx(self, path: Path) -> List[Dict[str, Any]]:
        doc = DocxDocument(str(path))
        parts = [p.text for p in doc.paragraphs if p.text.strip()]
        # 表格也取出来，用 | 分隔单元格
        for table in doc.tables:
            for row in table.rows:
                cells = [c.text.strip() for c in row.cells]
                line = " | ".join(c for c in cells if c)
                if line:
                    parts.append(line)
        text = _clean_text("\n".join(parts))
        if not text:
            raise ValueError("docx 中未提取到文本")
        return [self._make_doc(text, path, {})]

    def _load_pptx(self, path: Path) -> List[Dict[str, Any]]:
        pres = PptxPresentation(str(path))
        docs = []

        for slide_no, slide in enumerate(pres.slides, start=1):
            parts: List[str] = []

            def collect(shapes):
                for shape in shapes:
                    # ① 分组形状 → 递归进去
                    if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
                        collect(shape.shapes)
                        continue
                    # ② 文本框 / 占位符
                    if shape.has_text_frame and shape.text_frame.text.strip():
                        parts.append(shape.text_frame.text)
                    # ③ 表格
                    if getattr(shape, "has_table", False):
                        for row in shape.table.rows:
                            cells = [c.text.strip() for c in row.cells]
                            line = " | ".join(c for c in cells if c)
                            if line:
                                parts.append(line)

            collect(slide.shapes)

            # ④ 备注（讲稿）—— 有时比页面文字信息量更大
            try:
                notes = slide.notes_slide.notes_text_frame.text.strip()
                if notes:
                    parts.append(f"[备注] {notes}")
            except Exception:
                pass

            text = _clean_text("\n".join(parts))
            if not text:
                continue  # 纯图片幻灯片
            docs.append(self._make_doc(text, path, {"slide": slide_no}))

        if not docs:
            raise ValueError("pptx 中未提取到任何文本（可能全是图片）")
        return docs


    # ---------- 统一构造文档单元 ----------
    def _make_doc(self, content: str, path: Path, extra: Dict[str, Any]) -> Dict[str, Any]:
        meta = {
            "source": str(path),
            "filename": path.name,
            "suffix": path.suffix.lower(),
        }
        meta.update(extra)
        return {"content": content, "metadata": meta}


# ---------- 保持原接口不变 ----------
def get_documents(directory: str = None):
    # 默认语料目录基于本文件位置，不依赖当前工作目录
    if directory is None:
        DATA_DIR = Path(__file__).resolve().parent / "data" / "sample_documents"
    else:
        DATA_DIR = Path(directory)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ...
    loader = SimpleDocumentLoader(DATA_DIR)
    documents = loader.load()
    if loader.errors:
        print(f"\n⚠️ 有 {len(loader.errors)} 个文件加载失败：")
        for e in loader.errors:
            print(f"   - {e['file']}: {e['error']}")
    return documents


if __name__ == '__main__':
    docs = get_documents()
    total_chars = sum(len(d["content"]) for d in docs)
    print(f"\n成功加载 {len(docs)} 个文档单元，共 {total_chars} 字符")
    for i, d in enumerate(docs[:8], 1):
        meta = d["metadata"]
        page = f" 第{meta['page']}页" if "page" in meta else ""
        print(f"\n[{i}] {meta['filename']}{page}  ({len(d['content'])} 字符)")
        print(f"    {d['content'][:60]}...")