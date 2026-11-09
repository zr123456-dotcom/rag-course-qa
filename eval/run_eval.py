# -*- coding: utf-8 -*-
"""
RAG 评估脚本
用法（在项目根目录下执行）:
    python eval/run_eval.py
"""
import io
import sys
import json
import statistics
import contextlib
from pathlib import Path

# ---- 让脚本能 import 到项目根目录的模块 ----
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from t7_完整的RAG查询系统 import rag_system      # noqa: E402

QUESTIONS = ROOT / "eval" / "questions.json"
REPORT_DIR = ROOT / "eval" / "reports"

# 判定"拒答"的特征词
REFUSAL_MARKS = ("无法回答", "无法确定", "没有找到", "未提及",
                 "不包含", "不知道", "无法从资料")


# ---- 判定阈值（对应场景说明第 4、7 项）----
MIN_TOP1_SCORE = 0.52      # top1 相似度低于此值 → 视为"没检索到有效内容"
MAX_ANSWER_CHARS = 250     # 红线 4：答案正文字数 ≤ 250（去掉空白）

def load_questions():
    return json.loads(QUESTIONS.read_text(encoding="utf-8"))


def source_hit(retrieved_sources, expect_sources):
    """期望来源是否被命中（file 用子串匹配，给了 page/slide 则还要位置一致）"""
    if not expect_sources:
        return None                       # 不可答题不评检索
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


def run():
    qs = load_questions()
    print(f"共 {len(qs)} 道题，开始评估...\n")

    results = []
    for q in qs:
        # 屏蔽 RAG 流程里的步骤打印，让报告干净
        with contextlib.redirect_stdout(io.StringIO()):
           res = rag_system.query(q["question"])

        answer = res["answer"]
        sources = res["sources"]

        hit = source_hit(sources, q.get("expect_sources", []))
        cov = keyword_coverage(answer, q.get("expect_keywords", []))
        refused = is_refusal(answer)


        # ---- 新增：检索质量 + 答案长度（红线 4）----
        top1_score = max((s.get("score") or 0 for s in sources), default=0.0)
        retrieval_ok = top1_score >= MIN_TOP1_SCORE
        line_count = len(answer.splitlines())
        max_chars = q.get("max_chars", MAX_ANSWER_CHARS)
        char_count = len(answer.replace("\n", "").replace(" ", ""))
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
            "top1_score": top1_score,
            "retrieval_ok": retrieval_ok,
            "line_count": line_count,
            "char_count": char_count,
            "length_ok": length_ok,
            "passed": passed, "sources": sources,
        })

    report(results)


def report(results):
    n = len(results)
    answerable = [r for r in results if r["answerable"]]
    unanswerable = [r for r in results if not r["answerable"]]

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

    print("=" * 62)
    print(f"总题数             : {n}")
    print(f"检索命中率 (来源级) : {hit_rate:.1%}   ({sum(hits)}/{len(hits)})")
    print(f"关键词覆盖率 (均值) : {avg_cov:.1%}")
    print(f"top1 相似度 (均值)  : {avg_top1:.3f}   （门槛 {MIN_TOP1_SCORE}）")
    print(f"检索质量合格        : {retr_ok_n}/{len(answerable)}")
    print(f"答案长度合规 (≤{MAX_ANSWER_CHARS}字) : {len_ok_n}/{len(answerable)}")
    low_conf = [r for r in answerable if r.get("retrieval_ok") is False]
    print(f"低置信度题 (top1 <{MIN_TOP1_SCORE}) : "
          f"{len(low_conf)}/{len(answerable)}   ← 建议人工复核")
    if unanswerable:
        print(f"拒答正确率         : {refused_ok / len(unanswerable):.1%}"
              f"   ({refused_ok}/{len(unanswerable)})")
    print(f"整体通过           : {passed}/{n} = {passed / n:.1%}")
    print("=" * 62)

    if low_conf:
        print(f"\n⚠ 低置信度题（未判失败，但建议人工核实）：")
        for r in low_conf:
            print(f"  [{r['id']}] top1={r.get('top1_score'):.3f}  {r['question']}")

    print("\n按题型看：")
    cats = {}
    for r in results:
        cats.setdefault(r["category"], []).append(r)
    for cat, rs in cats.items():
        p = sum(1 for r in rs if r["passed"])
        print(f"  {cat:<14} {p}/{len(rs)}")

    fails = [r for r in results if not r["passed"]]
    if fails:
        print(f"\n未通过的题（{len(fails)} 道）：")
        for r in fails:
            print(f"\n  [{r['id']}] {r['question']}")
            print(f"    题型      : {r['category']}")
            print(f"    来源命中  : {r['source_hit']}")
            print(f"    关键词覆盖: {r['keyword_coverage']}")
            print(f"    答案字数  : {r.get('char_count')}   "
                  f"(长度合规: {r.get('length_ok')})")
            print(f"    top1 分数 : {r.get('top1_score') or 0:.3f}"
                  f"   (检索有效: {r.get('retrieval_ok')})")
            print(f"    答案行数  : {r.get('line_count')}")
            print(f"    是否拒答  : {r['refused']}")
            print(f"    答案      : {r['answer'][:100]}")
            print(f"    实际来源  : "
                  f"{[(s['file'][:12], s.get('page')) for s in r['sources']]}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / "latest.json"
    out.write_text(json.dumps({
                "summary": {
            "total": n, "hit_rate": hit_rate,
            "avg_keyword_coverage": avg_cov,
            "avg_top1_score": avg_top1,
            "retrieval_ok": retr_ok_n,
            "length_ok": len_ok_n,
            "refusal_accuracy": (refused_ok / len(unanswerable)
                                 if unanswerable else 0),
            "passed": passed,
        },
        "results": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    # 追加一行运行记录（保留历史，便于看趋势）
    import datetime
    with (REPORT_DIR / "runs.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "time": datetime.datetime.now().isoformat(timespec="seconds"),
            "total": n,
            "hit_rate": round(hit_rate, 4),
            "avg_cov": round(avg_cov, 4),
            "avg_top1": round(avg_top1, 4),
            "refusal_acc": round(refused_ok / len(unanswerable), 4) if unanswerable else 0,
            "passed": passed,
            "note": "15题全绿基线",
        }, ensure_ascii=False) + "\n")
    print(f"\n报告已保存: {out}")


if __name__ == "__main__":
    run()