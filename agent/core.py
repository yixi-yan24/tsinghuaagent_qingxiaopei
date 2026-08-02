import json, re
from collections.abc import Generator
from .memory import ShortTermMemory, LongTermMemory
from .llm_client import chat_completion, chat_completion_stream
from .tools import Tools
from .planner import CoursePlanner
from .prompts import SYSTEM_PROMPT
from .data_loader import load_programs


class TrainingPlanAgent:
    """The main agent orchestrator for Tsinghua Training Plan advising."""

    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com/v1"):
        self.api_key = api_key
        self.base_url = base_url

        # Load long-term memory (training program database)
        programs = load_programs()
        self.ltm = LongTermMemory(programs)

        # Initialize tools
        self.tools = Tools(self.ltm, api_key=api_key)

        # Initialize planner
        self.planner = CoursePlanner(api_key, base_url)

        # Initialize short-term memory (per-session, will be copied for each session)
        self._default_stm = ShortTermMemory()

    def create_session(self) -> "AgentSession":
        """Create a new conversation session."""
        return AgentSession(
            api_key=self.api_key,
            base_url=self.base_url,
            ltm=self.ltm,
            tools=self.tools,
            planner=self.planner
        )

class AgentSession:
    """A single conversation session with its own short-term memory."""

    # ── tool dispatch table (built once per class) ──────────────────────
    _TOOL_PARAM_MAP: dict[str, list[str]] = {
        "list_programs": [],
        "search_programs": ["keyword"],
        "get_program_detail": ["name"],
        "search_courses": ["keyword"],
        "get_course_detail": ["identifier"],
        "list_program_courses": ["program_name"],
        "check_requirements": ["major", "program_name"],
        "semantic_search": ["query"],
        "multi_agent_search": ["major", "interests", "grade"],
        "recommend_courses": ["major", "grade", "interests", "semester"],
        "generate_schedule_tool": ["major", "grade", "program_name", "completed_courses", "gpa", "goals", "target_semester"],
    }

    def __init__(
        self,
        api_key: str,
        base_url: str,
        ltm: LongTermMemory,
        tools: Tools,
        planner: CoursePlanner
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.ltm = ltm
        self.tools = tools
        self.planner = planner
        self.stm = ShortTermMemory()
        self.stm.add("system", SYSTEM_PROMPT)
        # Lazily-built tool block (cached so recursive _call_llm reuses it).
        self._tool_block: str | None = None

    # ── public API ──────────────────────────────────────────────────────

    def process_message(self, user_message: str, temperature: float = 0.3) -> str:
        """Process a user message and return the agent response."""
        self.stm.add("user", user_message)
        response = self._call_llm(temperature=temperature)
        self.stm.add("assistant", response)
        return response

    def process_message_stream(
        self, user_message: str, temperature: float = 0.3
    ) -> Generator[str, None, None]:
        """Like process_message but yields tokens for the **final** response.

        Internal tool-call loops are still non-streaming (the full response is
        needed to parse ACTION / PARAMS), but the last LLM turn uses true
        SSE streaming so the user sees tokens progressively.
        """
        self.stm.add("user", user_message)
        full_response: list[str] = []
        for token in self._call_llm_stream(temperature=temperature):
            full_response.append(token)
            yield token
        self.stm.add("assistant", "".join(full_response))

    def process_with_planning(self, major: str, grade: str, program_name: str) -> str:
        """Generate a course plan for a specific program."""
        plan = self.planner.generate_plan(major, grade, program_name, self.ltm)
        self.stm.add("user", f"请为{major}专业{grade}学生制定{program_name}修读计划")
        self.stm.add("assistant", plan)
        return plan

    def get_history(self) -> list[dict]:
        return self.stm.to_llm_format()

    def clear(self):
        self.stm.clear()
        self.stm.add("system", SYSTEM_PROMPT)

    # ── LLM calling ─────────────────────────────────────────────────────

    def _get_tool_block(self) -> str:
        """Build (once) and return the tool-use instruction block."""
        if self._tool_block is not None:
            return self._tool_block
        descs = self.tools.get_tool_descriptions()
        parts = [
            "\n\n你可以在回答前使用以下工具获取信息。如果需要使用工具，输出格式为：",
            "THOUGHT: <你的思考过程>",
            "ACTION: <工具名称>",
            'PARAMS: {"参数名": "参数值"}',
            "",
            "工具列表：",
        ]
        for t in descs:
            parts.append(f"- {t['name']}: {t['description']}")
            if t["parameters"]:
                for pname, pinfo in t["parameters"].items():
                    parts.append(f"  参数 {pname}: {pinfo.get('description', '')}")
        self._tool_block = "\n".join(parts)
        return self._tool_block

    def _call_llm(self, temperature: float = 0.3, tool_calls: int = 0) -> str:
        """Call DeepSeek API and handle tool use via prompt-based function calling.

        The system prompt inside STM is always kept clean — tool instructions
        are injected only into the outgoing request, so recursive calls never
        see a duplicated tool block.
        """
        messages = self.stm.to_llm_format()
        tool_block = self._get_tool_block()

        # Attach tool instructions to the system message (STM stays untouched).
        augmented_messages = []
        for msg in messages:
            if msg["role"] == "system":
                augmented_messages.append({
                    "role": "system",
                    "content": msg["content"] + "\n" + tool_block
                })
            else:
                augmented_messages.append(msg)

        content = chat_completion(
            self.api_key, self.base_url, augmented_messages,
            temperature=temperature, max_tokens=4096, timeout=90, retries=1,
        )

        # Tool-use loop (max 3 hops).
        tool_result = self._parse_tool_call(content)
        if tool_result:
            if tool_calls >= 3:
                return "抱歉，查询所需的工具调用次数过多。请缩小问题范围后重试。"
            tool_name, params = tool_result
            result = self._execute_tool(tool_name, params)
            self.stm.add("tool", result, tool_name=tool_name)
            return self._call_llm(temperature, tool_calls + 1)

        return self._clean_response(content)

    def _call_llm_stream(
        self, temperature: float = 0.3, tool_calls: int = 0
    ) -> Generator[str, None, None]:
        """Streaming variant of _call_llm.

        Uses true SSE streaming from DeepSeek.  Buffers the first few tokens
        to detect tool calls — if ``ACTION:`` appears early, the whole turn is
        consumed silently and the tool is executed.  Otherwise tokens are
        yielded progressively to the caller.
        """
        messages = self.stm.to_llm_format()
        tool_block = self._get_tool_block()

        augmented_messages = []
        for msg in messages:
            if msg["role"] == "system":
                augmented_messages.append({
                    "role": "system",
                    "content": msg["content"] + "\n" + tool_block
                })
            else:
                augmented_messages.append(msg)

        # Stream from DeepSeek — buffer just enough to detect tool calls.
        TOOL_DETECT_WINDOW = 200  # chars to inspect before deciding
        buffered: list[str] = []
        is_tool_call = False
        yielded = False

        for token in chat_completion_stream(
            self.api_key, self.base_url, augmented_messages,
            temperature=temperature, max_tokens=4096, timeout=90,
        ):
            if not is_tool_call:
                buffered.append(token)
                if "ACTION:" in "".join(buffered):
                    # Tool call detected — keep buffering silently.
                    is_tool_call = True
                elif len("".join(buffered)) >= TOOL_DETECT_WINDOW:
                    # Looks like a normal response — flush buffer progressively.
                    if not yielded:
                        yielded = True
                        for t in buffered:
                            yield t
                    else:
                        yield token
            else:
                buffered.append(token)

        if not yielded and not is_tool_call:
            # Response shorter than detection window — just flush everything.
            for t in buffered:
                yield t

        content = "".join(buffered)

        # Check if the LLM wants to use a tool
        if is_tool_call:
            tool_result = self._parse_tool_call(content)
            if tool_result:
                if tool_calls >= 3:
                    yield "抱歉，查询所需的工具调用次数过多。请缩小问题范围后重试。"
                    return
                tool_name, params = tool_result
                result = self._execute_tool(tool_name, params)
                self.stm.add("tool", result, tool_name=tool_name)
                yield from self._call_llm_stream(temperature, tool_calls + 1)
                return

    # ── helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _clean_response(content: str) -> str:
        """Strip THOUGHT: lines when the response does NOT contain a tool call.

        O(n) — scans lines twice at most.
        """
        lines = content.split("\n")
        # Pre-compute: are there ACTION lines?  If yes, keep everything for parsing.
        has_action = any(l.strip().startswith("ACTION:") for l in lines)
        if has_action:
            return content
        # No tool call — drop every THOUGHT: line.
        cleaned = [l for l in lines if not l.strip().startswith("THOUGHT:")]
        result = "\n".join(cleaned).strip()
        return result if result else content

    def _parse_tool_call(self, content: str):
        """Parse THOUGHT / ACTION / PARAMS from LLM output."""
        action_match = re.search(r"ACTION:\s*(\w+)", content)
        params_match = re.search(r"PARAMS:\s*(\{.*?\})", content, re.DOTALL)
        if action_match:
            tool_name = action_match.group(1)
            params = {}
            if params_match:
                try:
                    params = json.loads(params_match.group(1))
                except json.JSONDecodeError:
                    pass
            return tool_name, params
        return None

    def _execute_tool(self, tool_name: str, params: dict) -> str:
        """Dispatch *tool_name* to the matching method on self.tools."""
        tools = self.tools
        # Resolve the method once then call with only the parameters it expects.
        method = getattr(tools, tool_name, None)
        if method is None:
            return f"未知工具: {tool_name}"

        param_names = self._TOOL_PARAM_MAP.get(tool_name, [])
        kwargs = {p: params.get(p, "") for p in param_names}
        try:
            return method(**kwargs)
        except TypeError:
            # Fallback for no-arg tools (e.g. list_programs).
            return method()
