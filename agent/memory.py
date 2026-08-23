from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Message:
    role: str  # "user" | "assistant" | "system" | "tool"
    content: str
    tool_name: Optional[str] = None
    tool_call_id: Optional[str] = None   # 原生 function calling 的 tool_call id
    tool_calls: Optional[list] = None    # assistant 消息携带的原生 tool_calls
    reasoning_content: str = ""          # thinking mode 下 assistant 的思考内容


class ShortTermMemory:
    """Conversation history within a session."""

    def __init__(self, max_turns: int = 20):
        self.messages: list[Message] = []
        self.max_turns = max_turns
        self.max_message_characters = 20_000

    def add(self, role: str, content: str, tool_name: Optional[str] = None,
            tool_call_id: Optional[str] = None,
            tool_calls: Optional[list] = None,
            reasoning_content: str = ""):
        if len(content) > self.max_message_characters:
            content = content[:self.max_message_characters] + "\n[内容已截断]"
        self.messages.append(Message(
            role=role, content=content, tool_name=tool_name,
            tool_call_id=tool_call_id, tool_calls=tool_calls,
            reasoning_content=reasoning_content,
        ))
        if len(self.messages) > self.max_turns * 2:
            # Keep system prompt + recent history.
            keep = self.messages[:1] + self.messages[-(self.max_turns * 2 - 1):]
            # Trim to pairing boundaries: drop orphan tool messages at the
            # head and incomplete assistant tool_calls at the tail so the
            # history never breaks native function-calling pairing.
            start = 0
            while start < len(keep) and keep[start].role == "tool":
                start += 1
            end = len(keep)
            while end > start and keep[end - 1].role == "assistant" and keep[end - 1].tool_calls:
                end -= 1
            self.messages = keep[start:end] if end > start else keep

    def get_recent(self, n: int = 10) -> list[Message]:
        return self.messages[-n:]

    def get_all(self) -> list[Message]:
        return self.messages

    def clear(self):
        self.messages = []

    def to_llm_format(self) -> list[dict]:
        """Convert to OpenAI-format messages.

        Native function-calling pairs (assistant ``tool_calls`` followed by
        matching ``tool`` responses) are emitted as-is.  Any broken pair —
        orphan tool messages, tool_calls without responses, mismatched ids —
        is downgraded to plain text so the upstream API never rejects the
        history with a 400.
        """
        result: list[dict] = []
        i = 0
        n = len(self.messages)
        while i < n:
            msg = self.messages[i]
            if msg.role == "assistant" and msg.tool_calls:
                ids = {tc.get("id", "") for tc in msg.tool_calls if tc.get("id")}
                # Collect consecutive tool responses that follow this message.
                j = i + 1
                tool_msgs: list[Message] = []
                while j < n and self.messages[j].role == "tool":
                    tool_msgs.append(self.messages[j])
                    j += 1
                responded = {t.tool_call_id for t in tool_msgs}
                complete = bool(ids) and ids.issubset(responded) and len(tool_msgs) == len(ids)
                if complete:
                    entry = {
                        "role": "assistant",
                        "content": msg.content or None,
                        "tool_calls": msg.tool_calls,
                    }
                    if msg.reasoning_content:
                        # Thinking mode 要求把 assistant 的思考回传，否则多轮 400。
                        entry["reasoning_content"] = msg.reasoning_content
                    result.append(entry)
                    for t in tool_msgs:
                        result.append({
                            "role": "tool",
                            "tool_call_id": t.tool_call_id,
                            "content": t.content,
                        })
                else:
                    # Broken pairing — downgrade everything to plain text.
                    result.append({
                        "role": "assistant",
                        "content": msg.content or "（工具调用历史已省略）",
                    })
                    for t in tool_msgs:
                        result.append({
                            "role": "user",
                            "content": f"[工具 {t.tool_name or '工具'} 结果] {t.content[:2000]}",
                        })
                i = j
            elif msg.role == "tool":
                # Orphan tool message (no preceding assistant tool_calls).
                result.append({
                    "role": "user",
                    "content": f"[工具 {msg.tool_name or '工具'} 结果] {msg.content[:2000]}",
                })
                i += 1
            else:
                entry = {"role": msg.role, "content": msg.content}
                if msg.role == "assistant" and msg.reasoning_content:
                    entry["reasoning_content"] = msg.reasoning_content
                result.append(entry)
                i += 1
        return result


class LongTermMemory:
    """The parsed training program database — persistent knowledge."""

    def __init__(self, programs: list):
        self.programs = programs

    def find_program(self, name: str):
        from .data_loader import get_program_by_name
        return get_program_by_name(name, self.programs)

    def search(self, query: str):
        from .data_loader import search_programs
        return search_programs(query, self.programs)

    def list_all(self) -> list[str]:
        from .data_loader import get_all_program_names
        return get_all_program_names(self.programs)

    def get_overview(self) -> str:
        """Return a compact overview of all programs for the system prompt."""
        lines = []
        for m in self.programs:
            cap = f" ({m.department})" if m.department else ""
            lines.append(f"- {m.department} | {m.name}{cap}")
        return "\n".join(lines)
