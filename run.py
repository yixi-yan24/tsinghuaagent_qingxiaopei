#!/usr/bin/env python3
"""清华大学培养方案助手 - 启动入口"""

import os, sys, json

def _get_api_key() -> str:
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if key and key.startswith("sk-"):
        return key
    config_path = os.path.join(os.path.dirname(__file__), ".config")
    if os.path.isfile(config_path):
        with open(config_path, encoding="utf-8") as f:
            key = f.read().strip()
    if not key or not key.startswith("sk-"):
        print("错误: 请通过环境变量 DEEPSEEK_API_KEY 或 .config 文件设置有效的 DeepSeek API Key")
        sys.exit(1)
    return key

API_KEY = _get_api_key()


def _warmup_embedding():
    """Preload the semantic-search model in the background so the first
    conversation turn does not stall for ~15s inside a tool call."""
    try:
        from agent.embedding import _load_model
        _load_model()
        print("\n[i] 语义搜索模型已就绪。")
    except Exception:
        pass  # 加载失败不影响主流程，首次使用时再试


def run_cli():
    """Run the agent in interactive CLI mode (async flow via asyncio.run)."""
    import asyncio
    asyncio.run(_run_cli_async())


async def _run_cli_async():
    from agent.core import TrainingPlanAgent

    agent = TrainingPlanAgent(api_key=API_KEY)
    session = agent.create_session()

    print("=" * 60)
    print("  清华大学培养方案助手 v1.0")
    print("  输入 'quit' 退出 | 'clear' 清空对话 | 'plan' 生成修读计划")
    print("=" * 60)
    print()
    print("你好！我是你的培养方案助手。请问你的专业是什么？有什么课程规划方面的需求吗？")
    print()

    # 后台预加载词嵌入模型，避免首次语义搜索时长时间卡顿（静默加载）
    import threading
    threading.Thread(target=_warmup_embedding, daemon=True).start()

    while True:
        try:
            user_input = input("你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("再见！祝你学习顺利！")
            break
        if user_input.lower() == "clear":
            session.clear()
            print("对话已清空。")
            continue

        # Quick plan mode
        if user_input.lower().startswith("plan "):
            parts = user_input.split(maxsplit=3)
            if len(parts) >= 3:
                major = parts[1]
                grade = parts[2]
                program = parts[3] if len(parts) > 3 else ""
                if program:
                    print(f"\n助手 > 正在为您生成{program}修读计划...\n")
                    reply = session.process_with_planning(major, grade, program)
                    print(reply)
                    print()
                    continue
            print("用法: plan <专业> <年级> <培养方案名称>\n")
            continue

        print("\n助手 > ", end="", flush=True)
        await _stream_reply(session, user_input)
        print()
        print()


async def _stream_reply(session, user_input: str):
    """Consume the structured event stream and pretty-print it:
    reasoning in gray, tool status in yellow, final answer in default color."""
    import os
    if os.name == "nt":
        os.system("")  # 启用 Windows 终端 ANSI 转义序列
    GRAY, YELLOW, RESET = "\033[90m", "\033[33m", "\033[0m"

    in_thinking = False
    async for ev in session.process_message_events(user_input):
        if ev["type"] == "reasoning":
            if not in_thinking:
                print(f"{GRAY}💭 思考：{RESET}", end="", flush=True)
                in_thinking = True
            print(f"{GRAY}{ev['text']}{RESET}", end="", flush=True)
        elif ev["type"] == "tool":
            if in_thinking:
                print(RESET, end="", flush=True)
                in_thinking = False
            print(f"{YELLOW}\n🔍 正在调用工具 {ev['name']}…{RESET}", end="", flush=True)
        elif ev["type"] == "content":
            if in_thinking:
                print(RESET, end="", flush=True)
                in_thinking = False
                print("\n\n助手 > ", end="", flush=True)
            print(ev["text"], end="", flush=True)
    if in_thinking:
        print(RESET, end="", flush=True)


def run_api(host: str = "0.0.0.0", port: int = 8000):
    """Run the FastAPI server."""
    import uvicorn
    print(f"启动 API 服务器: http://{host}:{port}")
    print("API 文档: http://localhost:8000/docs")
    uvicorn.run("api.main:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="清华大学培养方案助手")
    parser.add_argument("--mode", choices=["cli", "api"], default="cli",
                        help="运行模式: cli (命令行交互) 或 api (HTTP服务)")
    parser.add_argument("--host", default="0.0.0.0", help="API 监听地址")
    parser.add_argument("--port", type=int, default=8000, help="API 监听端口")
    args = parser.parse_args()

    if args.mode == "api":
        run_api(args.host, args.port)
    else:
        run_cli()
