# -*- coding: utf-8 -*-
"""
RAG 评估脚本

用法（在任意目录执行均可）：
    python eval/run_eval.py

可选环境变量：
    EVAL_NOTE   写入 runs.jsonl 的备注，用于区分不同实验
                例：$env:EVAL_NOTE = "top_k=4 实验"
"""
import io
import os
import sys
import json
import time
import statistics
import contextlib
import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag_pipeline import rag_system      # noqa: E402

QUESTIONS = ROOT / "eval" / "questions.json"
REPORT_DIR = ROOT / "eval" / "reports"

REFUSAL_MARKS = ("无法回答", "无法确定", "没有找到", "未提及",
                 "不包含", "不知道", "无法从资料")

# ---- 判定阈值 ----
MIN_TOP1_SCORE = 0.52      # top1 相似度低于此值 → 低置信度（仅标记，不否决）
MAX_ANSWER_CHARS = 250     # 答案正文字数上限（去空白）

# ---- 重试 ----
MAX_RETRIES = 3


def load_questions():
    return json.loads(QUESTIONS.read_text(encoding="utf-8"))


def query_with_retry(question, retries=None):
    """调用 RAG；遇到服务端/网络错误时指数退避重试"""
    retries = retries or MAX_RETRIES
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                return rag_system.query(question)
        except Exception as exc:
            last_exc = exc
            if attempt == retries:
                break
            wait = 2 ** attempt          # 2s, 4s, 8s
            print(f"    [重试 {attempt}/{retries}] {type(exc).__name__}，{wait}s 后重试")
            time.sleep(wait)
    raise last_exc


def source_hit(retrieved_sources, expect_sources):
    """期望来源是否被命中（file 子串匹配；给了 page/slide 还要位置一致）"""
    if not expect_sources:
        return None
    for exp in expect_sources:
        for got in retrieved_sources:
            if exp["file"] not in got["file"]:
                continue
            ok = True
            if "page" in exp:
                ok = str(got.get("page")) == str(exp["page"])
            if "slide" in exp:
                ok = str(got.get("slide")) == str(exp["slide"])
            if ok:
                return True
    return False


def keyword_coverage(answer, keywords):
    if not keywords:
        return None
    hit = sum(1 for k in keywords if k.lower() in answer.lower())
    return hit / len(keywords)


def is_refusal(answer):
    return any(m in answer for m in REFUSAL_MARKS)


def error_record(q, exc):
    """API 调用失败时的占位记录，保证报告结构一致"""
    return {
        "id": q["id"], "category": q["category"], "question": q["question"],
        "answer": "", "answerable": q["answerable"],
        "source_hit": None, "keyword_coverage": None, "refused": False,
        "top1_score": 0.0, "retrieval_ok": False,
        "line_count": 0, "char_count": 0, "length_ok": False,
        "passed": False, "sources": [],
        "error": f"{type(exc).__name__}: {exc}",
    }


def save_partial(results):
    """增量落盘：后续即使崩溃，已完成的结果也在"""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "partial.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


def run():
    qs = load_questions()
    print(f"共 {len(qs)} 道题，开始评估...\n")

    results = []
    for idx, q in enumerate(qs, start=1):
        print(f"  [{idx}/{len(qs)}] {q['id']}  {q['question']}")
        try:
            res = query_with_retry(q["question"])
        except Exception as exc:
            print(f"    ✗ 跳过（{type(exc).__name__}: {exc}）")
            results.append(error_record(q, exc))
            save_partial(results)
            continue

        answer = res["answer"]
        sources = res["sources"]

        hit = source_hit(sources, q.get("expect_sources", []))
        cov = keyword_coverage(answer, q.get("expect_keywords", []))
        refused = is_refusal(answer)

        top1_score = max((s.get("score") or 0 for s in sources), default=0.0)
        retrieval_ok = top1_score >= MIN_TOP1_SCORE
        line_count = len(answer.splitlines())
        char_count = len(answer.replace("\n", "").replace(" ", ""))
        max_chars = q.get("max_chars", MAX_ANSWER_CHARS)
        length_ok = char_count <= max_chars

        if q["answerable"]:
            min_cov = q.get("min_coverage", 0.5)
            cov_ok = cov is not None and cov >= min_cov
            # 闸门：答案质量决定成败；检索分数只作风险信号
            passed = cov_ok and (hit is not False) and length_ok
        else:
            passed = refused

        results.append({
            "id": q["id"], "category": q["category"],
            "question": q["question"], "answer": answer,
            "answerable": q["answerable"], "source_hit": hit,
            "keyword_coverage": cov, "refused": refused,
            "top1_score": top1_score, "retrieval_ok": retrieval_ok,
            "line_count": line_count, "char_count": char_count,
            "length_ok": length_ok,
            "passed": passed, "sources": sources,
        })
        save_partial(results)

    report(results)


def report(results):
    n = len(results)
    errors = [r for r in results if r.get("error")]
    ok_results = [r for r in results if not r.get("error")]

    answerable = [r for r in ok_results if r["answerable"]]
    unanswerable = [r for r in ok_results if not r["answerable"]]

    hits = [r["source_hit"] for r in answerable if r["source_hit"] is not None]
    hit_rate = sum(hits) / len(hits) if hits else 0

    covs = [r["keyword_coverage"] for r in answerable
            if r["keyword_coverage"] is not None]
    avg_cov = statistics.mean(covs) if covs else 0

    refused_ok = sum(1 for r in unanswerable if r["refused"])
    passed = sum(1 for r in results if r["passed"])
    retr_ok_n = sum(1 for r in answerable if r.get("retrieval_ok"))
    len_ok_n = sum(1 for r in answerable if r.get("length_ok"))
    avg_top1 = (statistics.mean([r.get("top1_score", 0) for r in answerable])
                if answerable else 0)
    low_conf = [r for r in answerable if not r.get("retrieval_ok")]

    print("\n" + "=" * 62)
    print(f"总题数               : {n}")
    print(f"API 失败题数         : {len(errors)}")
    for e in errors:
        print(f"   - [{e['id']}] {e['error'][:70]}")
    print(f"检索命中率 (来源级)  : {hit_rate:.1%}   ({sum(hits)}/{len(hits)})")
    print(f"关键词覆盖率 (均值)  : {avg_cov:.1%}")
    print(f"top1 相似度 (均值)   : {avg_top1:.3f}   （门槛 {MIN_TOP1_SCORE}）")
    print(f"检索质量合格         : {retr_ok_n}/{len(answerable)}")
    print(f"答案长度合规 (≤{MAX_ANSWER_CHARS}字) : {len_ok_n}/{len(answerable)}")
    print(f"低置信度题 (top1<{MIN_TOP1_SCORE}) : {len(low_conf)}/{len(answerable)}")
    if unanswerable:
        print(f"拒答正确率           : {refused_ok / len(unanswerable):.1%}"
              f"   ({refused_ok}/{len(unanswerable)})")
    print(f"整体通过             : {passed}/{n} = {passed / n:.1%}")
    print("=" * 62)

    print("\n按题型看：")
    cats = {}
    for r in results:
        cats.setdefault(r["category"], []).append(r)
    for cat, rs in cats.items():
        p = sum(1 for r in rs if r["passed"])
        print(f"  {cat:<14} {p}/{len(rs)}")

    if low_conf:
        print("\n⚠ 低置信度题（未判失败，建议人工复核）：")
        for r in low_conf:
            print(f"  [{r['id']}] top1={r.get('top1_score'):.3f}  {r['question']}")

    fails = [r for r in results if not r["passed"]]
    if fails:
        print(f"\n未通过的题（{len(fails)} 道）：")
        for r in fails:
            print(f"\n  [{r['id']}] {r['question']}")
            print(f"    题型      : {r['category']}")
            if r.get("error"):
                print(f"    API 错误  : {r['error'][:80]}")
                continue
            print(f"    来源命中  : {r['source_hit']}")
            print(f"    关键词覆盖: {r['keyword_coverage']}")
            print(f"    答案字数  : {r.get('char_count')}  (合规: {r.get('length_ok')})")
            print(f"    答案行数  : {r.get('line_count')}")
            print(f"    top1 分数 : {r.get('top1_score'):.3f}  (有效: {r.get('retrieval_ok')})")
            print(f"    是否拒答  : {r['refused']}")
            print(f"    答案      : {(r['answer'] or '')[:100]}")
            print(f"    实际来源  : "
                  f"{[(s.get('file', '')[:12], s.get('page')) for s in r['sources']]}")

    # ---- 落盘 ----
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / "latest.json"
    out.write_text(json.dumps({
        "summary": {
            "total": n, "errors": len(errors),
            "hit_rate": hit_rate, "avg_keyword_coverage": avg_cov,
            "avg_top1_score": avg_top1,
            "retrieval_ok": retr_ok_n, "length_ok": len_ok_n,
            "refusal_accuracy": (refused_ok / len(unanswerable)
                                 if unanswerable else 0),
            "passed": passed,
        },
        "results": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    # 追加历史（note 可用环境变量覆盖）
    note = os.environ.get("EVAL_NOTE", f"{n}题基线")
    with (REPORT_DIR / "runs.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "time": datetime.datetime.now().isoformat(timespec="seconds"),
            "total": n, "errors": len(errors),
            "hit_rate": round(hit_rate, 4),
            "avg_cov": round(avg_cov, 4),
            "avg_top1": round(avg_top1, 4),
            "refusal_acc": (round(refused_ok / len(unanswerable), 4)
                            if unanswerable else 0),
            "passed": passed,
            "note": note,
        }, ensure_ascii=False) + "\n")

    print(f"\n报告已保存: {out}")
    print(f"历史已追加: {REPORT_DIR / 'runs.jsonl'}   note={note}")


if __name__ == "__main__":
    run()