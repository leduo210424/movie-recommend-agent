"""
System Prompt 量化对比实验：测试不同 Prompt 对 Agent Tool Calling 质量的影响。

指标：
  - 工具选择准确率：LLM 调用的工具是否与预期一致
  - 平均 Tool 调用次数：越少越高效
  - 成功率：最终是否返回了推荐结果

用法：
  python evaluate_prompts.py                    # 运行全部对比
  python evaluate_prompts.py --prompt minimal   # 只测试指定 prompt
  python evaluate_prompts.py --dry-run          # 仅打印测试用例，不发 API 调用
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ── Prompt 变体定义 ──

PROMPT_MINIMAL = """你是一个电影推荐助手。使用提供的工具为用户推荐电影。"""

PROMPT_STANDARD = """你是一个专业的电影推荐顾问 AI Agent。你的目标是根据用户的需求，使用可用的工具为用户找到最合适的电影推荐。

你拥有以下能力：
1. 获取用户的观影历史和偏好
2. 推荐热门/流行电影（冷启动场景）
3. 基于用户偏好的个性化推荐
4. 根据特定条件（类型、年份等）的精确搜索
5. 根据心情/氛围的推荐

你的工作流程：
1. 首先理解用户的查询需求
2. 根据需求选择合适的工具
3. 调用工具获取推荐结果
4. 根据结果进行综合分析和排序
5. 用中文给出有洞察的推荐和解释

推荐的关键原则：
- 多角度考虑（相似度、用户偏好、流行度）
- 给出清晰的推荐理由
- 如果用户信息不足，先获取用户资料
- 根据上下文智能选择推荐策略

在调用工具时，确保参数有效。在返回最终结果前，进行推理分析。"""

PROMPT_WORKFLOW_ONLY = """你是一个电影推荐助手。

严格按以下步骤工作：
1. 分析用户查询中的关键信息（用户ID、偏好类型、约束条件）
2. 从可用工具中选择最合适的 1-2 个工具
3. 调用工具获取结果
4. 用中文总结推荐

注意：先获取用户资料再推荐。"""

PROMPT_PRINCIPLES_ONLY = """你是一个电影推荐助手。

推荐原则（严格遵守）：
- 始终优先获取用户画像，了解其偏好后再推荐
- 对于有明确约束的查询（类型/年份），使用精确过滤
- 对于模糊/情绪化查询，使用语义个性化搜索
- 对于完全新用户，使用热门推荐兜底
- 每次只调用必要的工具，避免冗余调用
- 给出简洁有据的推荐理由"""

PROMPT_VARIANTS = {
    "minimal": PROMPT_MINIMAL,
    "standard": PROMPT_STANDARD,
    "workflow_only": PROMPT_WORKFLOW_ONLY,
    "principles_only": PROMPT_PRINCIPLES_ONLY,
}


# ── 标注测试用例 ──
# 格式: (user_id, query, top_k, expected_tools, description)
# expected_tools: 预期 LLM 应调用的工具名列表（顺序无关，但必须包含的）

TEST_CASES: List[Tuple[Optional[int], str, int, List[str], str]] = [
    # 冷启动场景
    (None, "推荐几部好看的电影", 5, ["search_cold_start"],
     "冷启动-无用户ID，直接热门推荐"),
    (None, "最近有什么好看的", 5, ["search_cold_start"],
     "冷启动-无用户ID，模糊请求"),

    # 精确过滤
    (1, "想看2010年后的科幻片", 5, ["search_by_filter"],
     "精确过滤-有年份+类型约束"),
    (1, "推荐几部2000年左右的喜剧电影", 5, ["search_by_filter"],
     "精确过滤-年份范围+类型"),

    # 个性化推荐（有用户ID + 模糊需求）
    (1, "想看轻松一点的电影", 5, ["search_by_preference"],
     "个性化-心情类模糊需求"),
    (1, "最近剧荒，推荐点好看的", 5, ["search_by_preference"],
     "个性化-模糊需求"),

    # 心情推荐
    (1, "想看让人放松的电影", 5, ["search_by_mood"],
     "心情推荐-明确心情词"),
    (1, "推荐几部刺激的动作片", 5, ["search_by_mood", "search_by_filter"],
     "混合-心情+类型"),
]


@dataclass
class PromptResult:
    prompt_name: str
    total_cases: int = 0
    tool_accuracy: float = 0.0
    avg_tool_calls: float = 0.0
    success_rate: float = 0.0
    total_tokens: int = 0
    details: List[Dict[str, Any]] = field(default_factory=list)


def evaluate_prompt(
    prompt_text: str,
    prompt_name: str,
    api_key: str,
    model: str = "qwen-plus",
    dry_run: bool = False,
    verbose: bool = True,
) -> PromptResult:
    """对一个 Prompt 变体运行全部测试用例"""
    result = PromptResult(prompt_name=prompt_name)

    if dry_run:
        print(f"\n  [DRY-RUN] Would test {len(TEST_CASES)} cases with prompt '{prompt_name}'")
        result.total_cases = len(TEST_CASES)
        return result

    try:
        from openai import OpenAI
    except ImportError:
        print("  [SKIP] openai not installed in current Python")
        return result

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    from src.react_agent import TOOLS as tools
    tool_names = {t["function"]["name"] for t in tools}

    result.total_cases = len(TEST_CASES)
    total_tool_calls = 0
    accurate_cases = 0
    successful_cases = 0

    for idx, (user_id, query, top_k, expected_tools, desc) in enumerate(TEST_CASES, 1):
        user_message = f"用户ID: {user_id or '无'}\n用户查询: {query}\n推荐数量: {top_k}\n请根据用户需求推荐电影。"

        messages = [
            {"role": "system", "content": prompt_text},
            {"role": "user", "content": user_message},
        ]

        if verbose:
            print(f"\n  [{idx}/{len(TEST_CASES)}] {desc}")
            print(f"    Query: {query}")
            print(f"    Expected: {expected_tools}")

        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                temperature=0.1,  # 低温度保证一致性
                max_tokens=1024,
                top_p=0.5,
            )

            msg = response.choices[0].message
            tool_calls = msg.tool_calls or []
            called_tools = []
            for tc in tool_calls:
                name = tc.function.name
                if name:
                    called_tools.append(name)

            total_tool_calls += len(called_tools)

            # 准确率：至少有一个预期工具被调用
            is_accurate = any(t in called_tools for t in expected_tools) if expected_tools else True
            is_success = len(called_tools) > 0

            if is_accurate:
                accurate_cases += 1
            if is_success:
                successful_cases += 1

            status = "OK" if is_accurate else "MISS"
            if verbose:
                print(f"    Called:  {called_tools}")
                print(f"    Status:  {status}")

            detail = {
                "case": desc,
                "query": query,
                "expected": expected_tools,
                "actual_tools": called_tools,
                "accurate": is_accurate,
                "success": is_success,
            }
            result.details.append(detail)

        except Exception as e:
            detail = {
                "case": desc,
                "query": query,
                "expected": expected_tools,
                "actual_tools": [],
                "accurate": False,
                "success": False,
                "error": str(e),
            }
            result.details.append(detail)
            if verbose:
                print(f"    Error: {e}")

        # API 限流保护
        time.sleep(0.3)

    result.tool_accuracy = accurate_cases / result.total_cases if result.total_cases > 0 else 0
    result.avg_tool_calls = total_tool_calls / result.total_cases if result.total_cases > 0 else 0
    result.success_rate = successful_cases / result.total_cases if result.total_cases > 0 else 0

    return result


def print_report(results: List[PromptResult]):
    print("\n" + "=" * 72)
    print("  System Prompt 对比实验报告")
    print("=" * 72)
    print(f"  {'Prompt':<20} {'准确率':>8} {'效率':>8} {'成功率':>8}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8}")

    for r in results:
        print(f"  {r.prompt_name:<20} {r.tool_accuracy:>7.1%} "
              f" {r.avg_tool_calls:>7.2f} {r.success_rate:>7.1%}")

    # 详情展开
    print(f"\n  --- 逐例详情 ---")
    for r in results:
        print(f"\n  [{r.prompt_name}]")
        for d in r.details:
            status = "✓" if d.get("accurate") else "✗"
            error = f" (ERROR: {d.get('error', '')[:40]})" if d.get("error") else ""
            print(f"    {status} {d['case']}")
            print(f"      expected={d['expected']}  called={d['actual_tools']}{error}")

    # 选出最佳
    if results:
        best = max(results, key=lambda r: r.tool_accuracy)
        print(f"\n  ★ 最佳 Prompt: {best.prompt_name} (准确率 {best.tool_accuracy:.1%})")
    print("=" * 72)


def main():
    parser = argparse.ArgumentParser(description="Compare System Prompt variants for Qwen Agent")
    parser.add_argument("--prompt", choices=list(PROMPT_VARIANTS.keys()),
                        help="Test only one specific prompt variant")
    parser.add_argument("--model", default="deepseek-chat", help="DeepSeek model to use")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print test cases without making API calls")
    args = parser.parse_args()

    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    if not api_key and not args.dry_run:
        print("请设置 DEEPSEEK_API_KEY 环境变量")
        print("  export DEEPSEEK_API_KEY=sk-xxxx  (Linux/Mac)")
        print("  set DEEPSEEK_API_KEY=sk-xxxx      (Windows)")
        sys.exit(1)

    print(f"测试用例数: {len(TEST_CASES)}")
    print(f"Prompt 变体数: {1 if args.prompt else len(PROMPT_VARIANTS)}")
    if api_key:
        print(f"API Key:     {api_key[:12]}...")

    results: List[PromptResult] = []

    variants_to_test = (
        {args.prompt: PROMPT_VARIANTS[args.prompt]}
        if args.prompt
        else PROMPT_VARIANTS
    )

    for name, prompt_text in variants_to_test.items():
        print(f"\n{'─'*60}")
        print(f"  测试 Prompt 变体: {name}")
        print(f"{'─'*60}")
        if args.dry_run:
            print(f"\n  Prompt 内容 ({len(prompt_text)} chars):")
            print(f"  {prompt_text[:200].replace(chr(10), chr(10)+'  ')}...")
        result = evaluate_prompt(
            prompt_text=prompt_text,
            prompt_name=name,
            api_key=api_key,
            model=args.model,
            dry_run=args.dry_run,
        )
        results.append(result)

    if not args.dry_run:
        print_report(results)


if __name__ == "__main__":
    main()
