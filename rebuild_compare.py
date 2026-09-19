"""从已有的原始结果文件重建多模型对照数据（不调用 API，零成本）。

背景：
    compare_models.py 每次运行都会覆盖 compare_results.json，
    而 compare_models.py 被同时跑了两次，导致"基础库"那份对照数据被难题库覆盖。

    但按「模型 + 题库」命名的原始结果文件（results_*.json）都还在，
    每道题都带 correct / latency / cost / tokens —— 统计量可以直接算出来，
    不需要重新调 API。

    这样做的另一个好处：报告里每个数字都能指回一个原始文件。

用法：
    python rebuild_compare.py            # 扫描目录下所有 results_*.json 并重建
"""
import os
import json
import glob
import statistics
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(BASE_DIR, "compare_rebuilt.json")

# 单价（元/百万 token，高峰价），与 pipeline.py 保持一致
PRICES = {
    "deepseek-flash":  (2.0, 8.0),
    "deepseek-v4-pro": (9.0, 27.0),
}


def parse_name(path):
    """从文件名解析出模型名与题库标签。

    results_deepseek-flash.json            -> ("deepseek-flash", "base")
    results_deepseek-flash_cases_hard.json -> ("deepseek-flash", "cases_hard")
    """
    stem = os.path.splitext(os.path.basename(path))[0]          # results_xxx
    body = stem[len("results_"):]
    for m in PRICES:
        if body == m:
            return m, "base"
        if body.startswith(m + "_"):
            return m, body[len(m) + 1:]
    return body, "unknown"


def summarize(items):
    ok = sum(1 for x in items if x.get("correct"))
    costs = [x["cost"] for x in items if x.get("cost") is not None]
    lats = [x["latency"] for x in items if x.get("latency") is not None]
    return {
        "n": len(items),
        "ok": ok,
        "accuracy": ok / len(items),
        "total_cost": sum(costs) if costs else None,
        "avg_cost": (sum(costs) / len(costs)) if costs else None,
        "avg_latency": statistics.mean(lats) if lats else None,
        "max_latency": max(lats) if lats else None,
        "per_item": {
            str(x["id"]): {"type": x["type"], "correct": bool(x.get("correct")),
                           "cost": x.get("cost"), "latency": x.get("latency")}
            for x in items
        },
    }


def main():
    files = sorted(glob.glob(os.path.join(BASE_DIR, "results_*.json")))
    groups = defaultdict(dict)
    for p in files:
        model, tag = parse_name(p)
        with open(p, encoding="utf-8") as f:
            items = json.load(f)
        groups[tag][model] = {"source": os.path.basename(p),
                              **summarize(items)}

    print(f"{'=' * 70}\n从已有原始文件重建的多模型对照（零成本，未调用 API）\n{'=' * 70}")
    for tag, models in groups.items():
        print(f"\n【题库：{tag}】")
        print(f"  {'模型':<20}{'来源文件':<40}{'准确率':<12}{'平均单题成本':<16}{'平均延迟'}")
        for m, s in models.items():
            print(f"  {m:<20}{s['source']:<40}"
                  f"{s['ok']}/{s['n']:<9}"
                  f"{s['avg_cost']:.6f} 元{'':<4}{s['avg_latency']:.2f}s")

        if len(models) == 2:
            ms = list(models)
            a, b = ms[0], ms[1]
            ca, cb = models[a]["avg_cost"], models[b]["avg_cost"]
            la, lb = models[a]["avg_latency"], models[b]["avg_latency"]
            print(f"  → 成本倍数（{b} ÷ {a}）：{cb / ca:.2f} 倍")
            print(f"  → 延迟（{a} vs {b}）：{la:.2f}s vs {lb:.2f}s（"
                  f"{a} 快 {abs(1 - la / lb) * 100:.0f}%）")
            if models[a]["accuracy"] == models[b]["accuracy"]:
                print(f"  → 准确率相同（都是 {models[a]['accuracy'] * 100:.1f}%），"
                      f"无精度差异")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"note": "由 rebuild_compare.py 从 results_*.json 重建",
                   "groups": groups}, f, ensure_ascii=False, indent=2)
    print(f"\n已写入: {OUT_PATH}")


if __name__ == "__main__":
    main()
