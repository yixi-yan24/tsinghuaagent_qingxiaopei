from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Message:
    role: str  # "user" | "assistant" | "system" | "tool"
    content: str
    tool_name: Optional[str] = None


class ShortTermMemory:
    """Conversation history within a session."""

    def __init__(self, max_turns: int = 20):
        self.messages: list[Message] = []
        self.max_turns = max_turns
        self.max_message_characters = 20_000

    def add(self, role: str, content: str, tool_name: Optional[str] = None):
        if len(content) > self.max_message_characters:
            content = content[:self.max_message_characters] + "\n[内容已截断]"
        self.messages.append(Message(role=role, content=content, tool_name=tool_name))
        if len(self.messages) > self.max_turns * 2:
            # Keep system prompt + recent history
            self.messages = self.messages[:1] + self.messages[-(self.max_turns * 2 - 1):]

    def get_recent(self, n: int = 10) -> list[Message]:
        return self.messages[-n:]

    def get_all(self) -> list[Message]:
        return self.messages

    def clear(self):
        self.messages = []

    def to_llm_format(self) -> list[dict]:
        """Convert to OpenAI-format messages."""
        result = []
        for msg in self.messages:
            if msg.role == "tool":
                result.append({"role": "user", "content": f"[工具 {msg.tool_name}] 结果: {msg.content}"})
            else:
                result.append({"role": msg.role, "content": msg.content})
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
