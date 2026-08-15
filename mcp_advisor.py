"""清华大学培养方案助手 — Claude Code MCP 桥接。

把部署在公网的 OpenAI 兼容服务包装成一个 MCP 工具，
使 Claude Code 会话中可以直接调用培养方案助手（查方案/选课/排课/修读计划）。

运行：python mcp_advisor.py   （需 pip install mcp httpx）
注册：claude mcp add advisor --scope project -- python C:/Users/yieza/Desktop/agentGame2/tsinghuaagent_qingxiaopei/mcp_advisor.py
"""
import os
import httpx
from mcp.server.fastmcp import FastMCP

# 公网服务地址与密钥（与服务器 .env 中 SERVICE_API_KEY 一致）
ADVISOR_URL = os.environ.get("ADVISOR_URL", "http://49.233.70.230:8000/v1/chat/completions")
ADVISOR_KEY = os.environ.get("ADVISOR_KEY", "sk-advisor-dev-key")
MODEL = "tsinghua-training-plan-advisor"

mcp = FastMCP("training-plan-advisor")


@mcp.tool()
def ask_training_advisor(question: str, session_id: str = "") -> str:
    """调用清华大学培养方案助手，回答培养方案、必修课、学分要求、排课、修读计划等问题。

    适合在用户问及清华本科培养方案、课程、选课、排课时调用。
    传入相同的 session_id 可以保持多轮对话上下文。
    """
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": question}],
        "stream": False,
    }
    if session_id:
        payload["user"] = session_id
    r = httpx.post(
        ADVISOR_URL,
        json=payload,
        headers={"Authorization": f"Bearer {ADVISOR_KEY}"},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


if __name__ == "__main__":
    mcp.run(transport="stdio")
