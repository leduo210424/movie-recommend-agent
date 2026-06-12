"""
调试脚本: 跟踪 ReAct Agent 的工具调用轨迹。

用法:
    export DEEPSEEK_API_KEY=sk-xxx
    python debug_trace.py --user-id 1 --query "轻松搞笑"

输出: 每一步的 iteration, LLM thought, tool_call, observation
"""
import argparse, json, logging, os, sys

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
sys.path.insert(0, ".")

logging.basicConfig(level=logging.WARNING)

from src.react_agent import ReActAgent, DeepSeekLLM, TOOLS

# ── 拦截 LLM 调用，打印完整交互 ──
class DebugLLM(DeepSeekLLM):
    def generate(self, messages, tools):
        print(f"\n{'='*60}")
        print(f"  LLM.generate() called — messages 共 {len(messages)} 条")
        print(f"{'='*60}")

        for i, msg in enumerate(messages):
            role = msg.get("role", "?")
            content = msg.get("content", "")
            if role == "system":
                print(f"  [{i}] SYSTEM ({len(content)} chars)")
            elif role == "user":
                text = content[:120].replace("\n", " ")
                print(f"  [{i}] USER: {text}...")
            elif role == "assistant":
                text = (content or "")[:120]
                print(f"  [{i}] ASSISTANT: {text}")
            else:
                print(f"  [{i}] {role}: {str(content)[:120]}")

        print(f"\n  Calling DeepSeek API...")
        response = super().generate(messages, tools)

        content = response.get("content", "") or ""
        tool_calls = response.get("tool_calls")

        print(f"  Response content ({len(content)} chars): {content[:200]}")
        if tool_calls:
            for tc in (tool_calls or []):
                func = tc.get("function", {})
                name = func.get("name", "?")
                args = func.get("arguments", {})
                print(f"  >>> ToolCall: {name}({json.dumps(args, ensure_ascii=False)})")
        else:
            print(f"  >>> NO tool_call — LLM 认为推理完成")

        return response


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user-id", type=int, default=1)
    parser.add_argument("--query", type=str, default="轻松搞笑")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    if not api_key:
        print("ERROR: DEEPSEEK_API_KEY 未设置!")
        print("请先: export DEEPSEEK_API_KEY=sk-xxx")
        return

    model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    print(f"Model: {model}")
    print(f"User: {args.user_id}, Query: {args.query}")

    llm = DebugLLM(api_key=api_key, model=model)
    agent = ReActAgent(llm=llm)

    result = agent.invoke(
        user_id=args.user_id,
        query=args.query,
        top_k=args.top_k,
        session_id="debug_trace",
    )

    print(f"\n{'='*60}")
    print(f"  最终结果")
    print(f"{'='*60}")
    print(f"  iterations: {result.get('iterations', '?')}")
    print(f"  route: {result.get('route', '?')}")
    print(f"  final_answer: {result.get('final_answer', '')[:300]}")
    print(f"  results count: {len(result.get('results', []))}")

    for i, r in enumerate(result.get("results", [])[:5]):
        print(f"\n  [{i+1}] {r.get('title')}")
        print(f"       user_sim={r.get('user_sim',0):.4f}  "
              f"rag_sim={r.get('rag_sim',0):.4f}  "
              f"pop={r.get('popularity',0):.4f}  "
              f"score={r.get('score',0):.4f}")


if __name__ == "__main__":
    main()
