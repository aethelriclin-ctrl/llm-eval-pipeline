"""多模型对照实验：同一套题、多个模型、重复 N 轮，比较【准确率 / 延迟 / 成本 / 稳定性】。

背景与目标：
    前面的实验发现 deepseek-flash 与 deepseek-v4-pro 在 17 道题上准确率完全相同，
    但 pro 的单价贵 4.5 倍、实测单题成本贵约 3.0 ~ 5.2 倍（多次跑测的区间）。

    单次跑测不足以支撑"两者无差异"这个结论（这是本项目反复踩过的坑），
    所以本脚本做两件事：
      1. 重复 N 轮，看准确率、延迟、成本是否稳定；
      2. 按任务类型做成本归因，给出"该用哪个模型"的建议。

    ⚠️ 注意本脚本的定位：它**不实现难度路由**——因为实测没有难度差异，
       按一个不存在的差异去分流是没有意义的。它做的是成本-效果分析。

用法：
    python compare_models.py                    # 2 个模型 × 5 轮
    python compare_models.py --rounds 3         # 指定轮数
    python compare_models.py --cases cases_hard.json
"""
import os
import sys
import json
import statistics
from collections import defaultdict

import pipeline as pl

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _out_path(cases_file, rounds):
    """按「题库 + 轮数 + 时间戳」命名，避免互相覆盖。

    为什么改这里：
        原先写死成 compare_results.json，**每跑一次就盖掉上一次**——
        "基础库 5 轮"那份数据就是这样被后来的难题库跑测冲掉的，
        文档里引用的数字一度没有文件可核。
    另外同时写一份 compare_results.json（_latest 副本），方便直接拿最近一次结果。
    """
    import time
    tag = os.path.splitext(os.path.basename(cases_file))[0] if cases_file else "base"
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return os.path.join(BASE_DIR, f"compare_{tag}_r{rounds}_{stamp}.json")


OUT_PATH = os.path.join(BASE_DIR, "compare_results.json")   # _latest 副本（兼容旧引用）
ROUNDS_DEFAULT = 5
MODELS = ["deepseek-flash", "deepseek-v4-pro"]


def run_one(model, cases_file, rounds):
    """把某个模型在某套题上跑 N 轮，返回每轮的汇总。

    直接把 pipeline.run 的 stdout 重定向掉，避免几十行输出淹没屏幕；
    只保留每轮的 results json 用于统计。
    """
    import io
    import contextlib

    per_round = []
    for r in range(1, rounds + 1):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            pl.run(limit=None, model=model, cases_file=cases_file)

        # run() 把结果写到 results_<模型>[_<题库>].json
        tag = ("_" + os.path.splitext(os.path.basename(cases_file))[0]) \
            if cases_file else ""
        res_path = os.path.join(BASE_DIR, f"results_{model}{tag}.json")
        with open(res_path, encoding="utf-8") as f:
            items = json.load(f)

        ok = sum(1 for x in items if x.get("correct"))
        cost = sum(x["cost"] for x in items if x.get("cost") is not None)
        lats = [x["latency"] for x in items if x.get("latency") is not None]
        per_round.append({
            "round": r,
            "n": len(items),
            "ok": ok,
            "accuracy": ok / len(items),
            "total_cost": cost,
            "avg_cost": cost / len(items),
            "avg_latency": statistics.mean(lats) if lats else None,
            "max_latency": max(lats) if lats else None,
            "items": items,
        })
        print(f"  [{model}] 第 {r}/{rounds} 轮：准确率 {ok}/{len(items)}，"
              f"成本 {cost:.6f} 元，平均延迟 "
              f"{(statistics.mean(lats) if lats else 0):.2f}s")
    return per_round


def summarize(model, per_round):
    """把一个模型的 N 轮结果压成统计量。"""
    accs = [x["accuracy"] for x in per_round]
    costs = [x["avg_cost"] for x in per_round]
    lats = [x["avg_latency"] for x in per_round]
    maxlats = [x["max_latency"] for x in per_round]

    # 逐题：N 轮里通过几次
    by_item = defaultdict(list)
    types = {}
    for rd in per_round:
        for it in rd["items"]:
            by_item[it["id"]].append(bool(it.get("correct")))
            types[it["id"]] = it["type"]

    return {
        "model": model,
        "rounds": len(per_round),
        "accuracy_mean": statistics.mean(accs),
        "accuracy_min": min(accs),
        "accuracy_max": max(accs),
        "accuracy_stable": len(set(accs)) == 1,
        "avg_cost_mean": statistics.mean(costs),
        "avg_cost_min": min(costs),
        "avg_cost_max": max(costs),
        "total_cost_sum": sum(x["total_cost"] for x in per_round),
        "avg_latency_mean": statistics.mean(lats),
        "avg_latency_max": max(lats),
        "max_latency_worst": max(maxlats),
        "per_item": {k: {"pass": sum(v), "n": len(v), "type": types[k]}
                     for k, v in sorted(by_item.items(), key=lambda kv: str(kv[0]))},
    }


def main(rounds, cases_file):
    label = cases_file or "cases.json（默认 11 题）"
    print(f"{'=' * 66}\n多模型对照实验\n模型：{MODELS}\n题库：{label}\n轮数：{rounds}\n{'=' * 66}")

    summaries = {}
    for m in MODELS:
        print(f"\n---- 跑 {m} ----")
        summaries[m] = summarize(m, run_one(m, cases_file, rounds))

    # ============ 汇总表 ============
    print("\n" + "=" * 66)
    print("【总览】")
    print(f"  {'模型':<20}{'准确率(均值/最低)':<22}{'平均单题成本':<16}{'平均延迟'}")
    for m in MODELS:
        s = summaries[m]
        print(f"  {m:<20}"
              f"{s['accuracy_mean'] * 100:5.1f}% / {s['accuracy_min'] * 100:5.1f}%"
              f"{'':<8}{s['avg_cost_mean']:.6f} 元{'':<4}"
              f"{s['avg_latency_mean']:.2f}s")

    # ============ 稳定性 ============
    print("\n" + "=" * 66)
    print("【稳定性】（重复跑测的意义：单次结果不能代表稳定水平）")
    for m in MODELS:
        s = summaries[m]
        print(f"  {m}:")
        print(f"    准确率是否每轮一致：{'是' if s['accuracy_stable'] else '否'}"
              f"（区间 {s['accuracy_min'] * 100:.1f}% ~ {s['accuracy_max'] * 100:.1f}%）")
        print(f"    单题成本波动：{s['avg_cost_min']:.6f} ~ {s['avg_cost_max']:.6f} 元"
              f"（{s['avg_cost_max'] / s['avg_cost_min']:.1f} 倍）")
        print(f"    延迟：平均 {s['avg_latency_mean']:.2f}s，"
              f"最慢一轮的平均 {s['avg_latency_max']:.2f}s，"
              f"单题最慢 {s['max_latency_worst']:.2f}s")

    # ============ 逐题对比 ============
    print("\n" + "=" * 66)
    print("【逐题对比】（N 轮里通过几次）")
    a, b = MODELS
    print(f"  {'题目':<8}{'类型':<16}{a:<22}{b}")
    all_ids = sorted(set(summaries[a]["per_item"]) | set(summaries[b]["per_item"]),
                     key=str)
    for i in all_ids:
        pa = summaries[a]["per_item"].get(i, {"pass": 0, "n": 0, "type": "?"})
        pb = summaries[b]["per_item"].get(i, {"pass": 0, "n": 0, "type": "?"})
        mark = "  ← 有差异" if pa["pass"] != pb["pass"] else ""
        print(f"  {str(i):<8}{pa['type']:<16}"
              f"{pa['pass']}/{pa['n']:<18}{pb['pass']}/{pb['n']}{mark}")

    # ============ 成本-效果结论 ============
    print("\n" + "=" * 66)
    print("【成本-效果结论】")
    sa, sb = summaries[a], summaries[b]
    ratio = sb["avg_cost_mean"] / sa["avg_cost_mean"]
    print(f"  单价与实测成本：{b} 的平均单题成本是 {a} 的 {ratio:.1f} 倍")
    if abs(sa["accuracy_mean"] - sb["accuracy_mean"]) < 1e-9:
        print(f"  两者的准确率均值相同（都是 {sa['accuracy_mean'] * 100:.1f}%）")
        print(f"  → 在本题库覆盖的任务类型上，{b} 没有可测量的精度优势；")
        print(f"     最优策略是全部使用 {a}，可节省约 "
              f"{(1 - sa['avg_cost_mean'] / sb['avg_cost_mean']) * 100:.0f}% 的费用。")
    else:
        diff = (sb["accuracy_mean"] - sa["accuracy_mean"]) * 100
        print(f"  准确率差异：{diff:+.1f} 个百分点")
        extra = sb["avg_cost_mean"] - sa["avg_cost_mean"]
        print(f"  → 为提升 {diff:.1f} 个百分点，每题多付 {extra:.6f} 元；"
              f"是否值得取决于业务对精度的要求。")

    # ============ 落盘 ============
    payload = {"rounds": rounds, "cases_file": cases_file, "summaries": summaries}
    stamped = _out_path(cases_file, rounds)
    with open(stamped, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n原始结果已写入: {stamped}")
    print(f"（同时更新最近一次副本: {OUT_PATH}）")


if __name__ == "__main__":
    rounds = ROUNDS_DEFAULT
    if "--rounds" in sys.argv:
        rounds = int(sys.argv[sys.argv.index("--rounds") + 1])
    cases = None
    if "--cases" in sys.argv:
        cases = sys.argv[sys.argv.index("--cases") + 1]
    if not pl.API_KEY:
        print("未读取到环境变量 DEEPSEEK_API_KEY")
        raise SystemExit(1)
    main(rounds, cases)
