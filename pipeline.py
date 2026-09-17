r"""
大模型评测流水线

用法（PowerShell）：
    1) 设置 API Key：
           $env:DEEPSEEK_API_KEY="你的key"        # 仅当前窗口有效
           setx DEEPSEEK_API_KEY "你的key"        # 永久生效，需重开窗口
    2) 运行：
           cd <项目目录>
           python pipeline.py            # 跑全部题目
           python pipeline.py --limit 5  # 只跑前 5 题（调试用）

设计要点见 README.md。
"""

import os
import sys
import json
import re
import time
from openai import OpenAI

# ============ 配置 ============
MODEL = "deepseek-chat"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CASES_PATH = os.path.join(BASE_DIR, "cases.json")
RESULTS_PATH = os.path.join(BASE_DIR, "results.json")

# 单价（元 / 百万 token），默认留空。
# 需要成本维度时，从官方定价页取真实值填入，并在提交信息里注明取值日期。
# 留空时 estimate_cost() 返回 None，汇总里显示"未填单价"——不猜、不编。
PRICE_INPUT_PER_M = None
PRICE_OUTPUT_PER_M = None

API_KEY = os.environ.get("DEEPSEEK_API_KEY")

_client = None


def get_client():
    """延迟创建客户端。

    不在模块加载时创建，是为了在 Key 缺失时给出可读的中文提示，
    而不是让 SDK 直接抛出英文的 credentials 异常。
    """
    global _client
    if _client is None:
        if not API_KEY:
            raise RuntimeError(
                "未读取到环境变量 DEEPSEEK_API_KEY。\n"
                "  PowerShell 临时设置：$env:DEEPSEEK_API_KEY=\"你的key\"\n"
                "  永久设置：setx DEEPSEEK_API_KEY \"你的key\"（需重开窗口）"
            )
        _client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")
    return _client


def ask_model(prompt, model_name=MODEL):
    """向模型发一道题。

    返回 (回答文本, 输入token, 输出token, 延迟秒, 错误信息)。
    错误信息非空表示本次请求失败，调用方需要单独处理，
    因为请求失败属于评测基础设施故障，不应计入模型能力。
    """
    start = time.time()
    try:
        resp = get_client().chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,  # 评测需要可重复性
        )
        latency = time.time() - start
        text = resp.choices[0].message.content
        usage = resp.usage
        return text, usage.prompt_tokens, usage.completion_tokens, latency, None
    except Exception as e:
        return "", 0, 0, time.time() - start, str(e)


def estimate_cost(in_tokens, out_tokens):
    """按官方单价估算成本（元）。单价未填时返回 None，不做任何假设。"""
    if PRICE_INPUT_PER_M is None or PRICE_OUTPUT_PER_M is None:
        return None
    return (in_tokens / 1_000_000 * PRICE_INPUT_PER_M
            + out_tokens / 1_000_000 * PRICE_OUTPUT_PER_M)


# ============ 评分器 ============
# 统一约定：每个评分器返回 (是否正确, 原因标签)。
# 原因标签会进入失败归因表，取值限定为：
#     "判对" / "能力不足" / "格式不合规" / "评分器误判" / "题目歧义"
# 分开标记的意义：只有"能力不足"才是模型的账，
# "格式不合规""评分器误判""题目歧义"都属于评测侧问题，会影响结论的可信度。


def grader_exact(answer, standard):
    """精确匹配。

    先只去首尾空白；若此时相等即判对。
    若不等，再剥掉常见标点比较一次——若此时相等，说明内容正确、
    只是格式不合规，标记为"格式不合规"，与真正的答错区分开。
    """
    if answer is None:
        return False, "能力不足"

    got = answer.strip()
    want = standard.strip()

    if got == want:
        return True, "判对"

    strip_punct = lambda s: re.sub(r"[。，、！？；：\"'.,!?;: \t\n]", "", s)
    if strip_punct(got) == strip_punct(want):
        return False, "格式不合规"

    return False, "能力不足"


def grader_contains_any(answer, keywords):
    """命中任一关键词即算通过，用于答案表述多样的事实问答。"""
    if answer is None:
        return False, "能力不足"

    text = answer.strip()
    for kw in keywords:
        if kw in text:
            return True, "判对"
    return False, "能力不足"


def grader_contains_all(answer, keywords):
    """所有关键词组都命中才算通过，用于要求答全的题目。

    keywords 支持两种写法：
        平铺：["404", "500"]                  → 每个词都必须出现
        分组：[["404"], ["不存在", "未找到"]]  → 每组至少命中一个词
    分组写法用于容纳同义表述：模型答"资源未找到"而答案钥写"不存在"时，
    初版平铺结构会误判为失败（假阴性），分组结构解决了这个问题。
    """
    if answer is None:
        return False, "能力不足"

    text = answer.strip()
    for kw in keywords:
        if isinstance(kw, list):
            if not any(word in text for word in kw):
                return False, "能力不足"
        else:
            if kw not in text:
                return False, "能力不足"
    return True, "判对"


def grader_json(answer, standard_obj):
    """JSON 格式约束评分。

    需要处理模型三种常见的不合规输出：
      1. 用 ```json ... ``` 代码块包裹（题目已明确要求不要）
      2. 在 JSON 前后附加解释文字
      3. Python 中 True == 1 成立，值类型必须单独校验，
         否则 {"age": true} 会被误判为等于 {"age": 1}

    归因规则：
      解析失败但剥掉包裹后确实拿到了类 JSON 内容 → "格式不合规"
      内容中不存在 JSON                          → "能力不足"
      解析成功但字段或值不符                      → "能力不足"
    """
    if answer is None:
        return False, "能力不足"

    text = answer.strip()

    # 剥掉 markdown 代码块标记，同时记录它是否出现过（归因时需要）
    had_fence = "```" in text
    text = re.sub(r"```[a-zA-Z]*\n?", "", text).replace("```", "").strip()

    # 截取第一个 { 到最后一个 } 之间的内容
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return False, "能力不足"
    candidate = text[start:end + 1]

    try:
        parsed = json.loads(candidate)
    except Exception:
        looks_like_json = '"' in candidate and ":" in candidate
        return False, ("格式不合规" if (had_fence or looks_like_json) else "能力不足")

    if not isinstance(parsed, dict):
        return False, "能力不足"
    if parsed.keys() != standard_obj.keys():
        return False, "能力不足"

    for k, v in standard_obj.items():
        got = parsed[k]
        if type(got) is not type(v):   # 挡住 True == 1 这类隐式相等
            return False, "格式不合规"
        if got != v:
            return False, "能力不足"

    return True, "判对"


def grader_code(answer, standard_expr):
    """代码题：执行模型给出的代码，再验证其行为。

    判分依据是行为而非文本：模型可能写出格式漂亮但无法运行的代码，
    也可能写出风格粗糙但完全正确的代码，因此用执行结果判分。

    standard_expr 来自题库，例如 "add(2, 3) == 5"。

    安全说明：exec/eval 会真正执行模型生成的代码。本地评测可以接受；
    生产环境必须放入子进程或容器隔离，限制文件系统与网络访问并设置超时。
    """
    if answer is None:
        return False, "能力不足"

    raw = answer.strip()
    had_fence = "```" in raw
    code = re.sub(r"```[a-zA-Z]*\n?", "", raw).replace("```", "").strip()

    # 独立的命名空间，避免模型代码中的变量污染主程序环境
    ns = {}

    try:
        exec(code, ns)
    except Exception:
        return False, "能力不足"

    try:
        ok = eval(standard_expr, ns)
    except Exception:
        return False, "能力不足"

    if ok is True:
        return True, "判对"

    # 行为不符：若它还没遵守输出格式要求，单独标记，便于归因区分
    return False, ("格式不合规" if had_fence else "能力不足")


GRADERS = {
    "exact": grader_exact,
    "contains_any": grader_contains_any,
    "contains_all": grader_contains_all,
    "json": grader_json,
    "code": grader_code,
}


# ============ 主流程 ============

def run(limit=None):
    with open(CASES_PATH, encoding="utf-8") as f:
        cases = json.load(f)
    if limit:
        cases = cases[:limit]

    results = []
    for c in cases:
        print(f"\n[{c['id']}/{len(cases)}] ({c['type']}) {c['prompt'][:40]}...")
        text, tin, tout, lat, err = ask_model(c["prompt"])
        if err:
            print(f"  请求失败: {err}")
            results.append({**c, "answer": None, "error": err, "correct": False})
            continue

        grader = GRADERS[c["check"]]
        correct, reason = grader(text, c["answer"])

        # 人工标注优先于自动判分。
        # known_issue 由人工复核后写入题库，用于标记题目本身存在问题的样本。
        # 纯自动判分无法区分"模型不会"和"题目有毛病"，因此必须保留人工校准入口。
        if c.get("known_issue"):
            reason = c["known_issue"]

        cost = estimate_cost(tin, tout)

        print(f"  回答: {text!r}")
        print(f"  判分: {'✅' if correct else '❌'}  原因: {reason}")
        print(f"  延迟: {lat:.2f}s  token: {tin}+{tout}  "
              f"成本: {cost if cost is not None else '未填单价'}")

        results.append({
            "id": c["id"], "type": c["type"], "prompt": c["prompt"],
            "answer": text, "correct": correct, "reason": reason,
            "latency": round(lat, 3), "tokens_in": tin, "tokens_out": tout,
            "cost": cost,
        })

    # ---- 总体 ----
    print("\n" + "=" * 60)
    print(f"模型: {MODEL}   题目数: {len(results)}")
    ok = sum(1 for r in results if r.get("correct"))
    print(f"准确率: {ok}/{len(results)} = {ok / len(results) * 100:.1f}%")

    # ---- 分类统计 ----
    # 总分会把不同能力维度混在一起：一个模型总分 70%，可能是样样 70%，
    # 也可能是部分维度满分、部分维度全错——两者的结论完全不同。
    by_type = {}
    for r in results:
        s = by_type.setdefault(r["type"], {"total": 0, "ok": 0})
        s["total"] += 1
        if r.get("correct"):
            s["ok"] += 1

    print("\n" + "-" * 60)
    print("【分类准确率】")
    for t, s in sorted(by_type.items(), key=lambda kv: -kv[1]["total"]):
        print(f"  {t:<8} {s['ok']}/{s['total']}  {s['ok'] / s['total'] * 100:5.1f}%")

    # ---- 失败归因 ----
    fails = [r for r in results if not r.get("correct")]
    print("\n" + "-" * 60)
    if not fails:
        print("【失败归因】无失败")
    else:
        print(f"【失败归因】共 {len(fails)} 道失败")
        by_reason = {}
        for r in fails:
            by_reason.setdefault(r.get("reason") or "未标注", []).append(r["id"])
        for reason, ids in sorted(by_reason.items(), key=lambda kv: -len(kv[1])):
            print(f"  {reason:<8} {len(ids)} 道   题目 id: {ids}")

        model_fault = sum(len(v) for k, v in by_reason.items() if k == "能力不足")
        grader_fault = sum(len(v) for k, v in by_reason.items()
                           if k in ("题目歧义", "评分器误判", "格式不合规"))
        effective_ok = ok + grader_fault
        print(f"\n  → 归因给【模型能力】的失败: {model_fault} 道")
        print(f"  → 归因给【题目/评分器】的失败: {grader_fault} 道")
        print(f"  → 原始准确率:   {ok}/{len(results)} = {ok / len(results) * 100:.1f}%")
        print(f"  → 修正后准确率: {effective_ok}/{len(results)} = "
              f"{effective_ok / len(results) * 100:.1f}%   （把评测侧问题造成的失败还回去）")

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n原始结果已写入: {RESULTS_PATH}")


if __name__ == "__main__":
    if not API_KEY:
        print("未读取到环境变量 DEEPSEEK_API_KEY。")
        print("  PowerShell 临时设置：$env:DEEPSEEK_API_KEY=\"你的key\"")
        print("  永久设置：setx DEEPSEEK_API_KEY \"你的key\"（需重开窗口）")
        sys.exit(1)
    limit = None
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])
    run(limit)
