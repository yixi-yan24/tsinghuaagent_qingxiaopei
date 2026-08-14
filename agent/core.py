import json, re
from collections.abc import Generator
from .memory import ShortTermMemory, LongTermMemory
from .llm_client import chat_completion, chat_completion_stream
from .tools import Tools
from .planner import CoursePlanner
from .prompts import SYSTEM_PROMPT
from .data_loader import load_programs

# ── scaffold markers the LLM emits for prompt-based tool calling ────────
_SCAFFOLD_PREFIXES = ("THOUGHT:", "ACTION:", "PARAMS:")
_SCAFFOLD_BARE = ("无", "none", "None")


def _is_scaffold_line(line: str) -> bool:
    """True if a line looks like THOUGHT/ACTION/PARAMS scaffold (or a bare
    ``无`` meaning "no tool needed") that must be hidden from the user."""
    s = line.strip()
    return s.startswith(_SCAFFOLD_PREFIXES) or s in _SCAFFOLD_BARE


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

    # Max tool hops before forcing a direct answer (a legitimate multi-step
    # flow usually needs 2–4; beyond that the loop is not converging).
    MAX_TOOL_HOPS = 5

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
        "generate_schedule": ["major", "grade", "program_name", "completed_courses", "gpa", "goals", "target_semester"],
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
        # Tool calls already executed within the current user turn (used to
        # detect non-converging loops, e.g. repeated identical searches).
        self._seen_tool_calls: set[tuple[str, str]] = set()

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

        # Tool-use loop.  Record the assistant's own ACTION turn before the
        # tool result, exactly like native OpenAI tool-calls, so the model can
        # see which calls it already made and does not repeat them.
        tool_result = self._parse_tool_call(content)
        if tool_result:
            tool_name, params = tool_result
            if tool_calls == 0:
                self._seen_tool_calls.clear()
            call_key = (tool_name, json.dumps(params, ensure_ascii=False, sort_keys=True))
            # Loop is not converging (repeated identical call, or budget used
            # up) — force a direct answer instead of erroring out.
            if tool_calls >= self.MAX_TOOL_HOPS or call_key in self._seen_tool_calls:
                return self._force_final_answer(temperature)
            self._seen_tool_calls.add(call_key)
            result = self._execute_tool(tool_name, params)
            self.stm.add("assistant", content)
            self.stm.add("tool", result, tool_name=tool_name)
            return self._call_llm(temperature, tool_calls + 1)

        return self._clean_response(content) or "抱歉，暂时未能生成有效回答。请换个问法后重试。"

    @staticmethod
    def _extract_answer_block(content: str) -> str:
        """Pull the model's answer out of a ```answer ... ``` code block.

        In forced-answer mode the model is told to wrap its reply in such a
        block, which makes scaffold/thinking material easy to discard.
        """
        match = re.search(r"```answer\s*\n(.*?)\n```", content, re.DOTALL)
        if match:
            return match.group(1).strip()
        match = re.search(r"```\s*\n(.*?)\n```", content, re.DOTALL)
        return match.group(1).strip() if match else ""

    def _force_final_answer(self, temperature: float = 0.3) -> str:
        """Last resort: ask the model to answer directly, with no tools available.

        Context is rebuilt from scratch (question + tool results only) so the
        model has nothing in history to mimic and cannot emit THOUGHT/ACTION.
        The reply must be wrapped in a ```answer ... ``` fence for robust
        extraction.
        """
        user_question = ""
        tool_results: list[str] = []
        for msg in self.stm.messages:
            if msg.role == "user":
                user_question = msg.content
            elif msg.role == "tool":
                tool_results.append(f"[工具 {msg.tool_name}] 结果:\n{msg.content}")

        final_messages = [{
            "role": "system",
            "content": SYSTEM_PROMPT + (
                "\n\n以下是本次查询过程中收集到的工具结果。"
                "请直接根据这些结果回答用户的问题，给出完整、具体的回答。"
                "必须把最终回答完整地放在 ```answer 和 ``` 代码块之间，"
                "代码块之外不要输出任何其他内容。"
                "不要输出思考过程，不要出现 THOUGHT / ACTION / PARAMS 等标记，"
                "也不要再尝试搜索或调用工具。"
                "如果结果中不包含用户需要的信息，请如实告知资料未被收录。"
            ),
        }]
        if user_question:
            final_messages.append({"role": "user", "content": user_question})
        for r in tool_results:
            final_messages.append({"role": "user", "content": r})

        content = chat_completion(
            self.api_key, self.base_url, final_messages,
            temperature=temperature, max_tokens=4096, timeout=90, retries=1,
        )
        result = self._extract_answer_block(content) or self._clean_response(content)
        if not result:
            # Model failed to use the fence — one strict retry.
            final_messages.append({"role": "user", "content": "请把回答放在 ```answer 和 ``` 之间，直接输出回答正文。"})
            content = chat_completion(
                self.api_key, self.base_url, final_messages,
                temperature=temperature, max_tokens=4096, timeout=90, retries=1,
            )
            result = self._extract_answer_block(content) or self._clean_response(content)
        if not result:
            return "抱歉，暂时未能生成有效回答。您可以换个问法，或缩小问题范围后重试。"
        return result

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
        pending_line = ""
        skip_until_blank = False

        def emit(tokens: list[str]) -> Generator[str, None, None]:
            """Yield tokens line by line, dropping scaffold material.

            A ``THOUGHT:`` line often wraps over several lines; everything up
            to the blank line that ends the thought block is hidden too.
            """
            nonlocal pending_line, skip_until_blank
            for tok in tokens:
                pending_line += tok
                while "\n" in pending_line:
                    line, pending_line = pending_line.split("\n", 1)
                    s = line.strip()
                    if _is_scaffold_line(line):
                        skip_until_blank = True
                        continue
                    if skip_until_blank:
                        if s:
                            continue  # still inside the thought block
                        skip_until_blank = False
                        continue  # blank line ends the block
                    yield line + "\n"

        def flush_tail() -> Generator[str, None, None]:
            nonlocal pending_line
            if not pending_line:
                return
            line, pending_line = pending_line, ""
            if not _is_scaffold_line(line):
                yield line

        for token in chat_completion_stream(
            self.api_key, self.base_url, augmented_messages,
            temperature=temperature, max_tokens=4096, timeout=90,
        ):
            if is_tool_call:
                buffered.append(token)
                continue
            if not yielded:
                buffered.append(token)
                if "ACTION:" in "".join(buffered):
                    # Tool call detected — keep buffering silently.
                    is_tool_call = True
                elif len("".join(buffered)) >= TOOL_DETECT_WINDOW:
                    # Looks like a normal response — stream it (scaffold-filtered).
                    yielded = True
                    yield from emit(buffered)
                    buffered.clear()
            else:
                # Already streaming a normal answer — keep going even if a
                # stray "ACTION:" appears; it is just part of the text.
                yield from emit([token])

        if is_tool_call:
            content = "".join(buffered)
        elif yielded:
            content = "".join(buffered)
            yield from flush_tail()
        else:
            # Response shorter than detection window — flush everything.
            content = "".join(buffered)
            yield from emit(buffered)
            yield from flush_tail()

        # Check if the LLM wants to use a tool
        if is_tool_call:
            tool_result = self._parse_tool_call(content)
            if tool_result:
                tool_name, params = tool_result
                if tool_calls == 0:
                    self._seen_tool_calls.clear()
                call_key = (tool_name, json.dumps(params, ensure_ascii=False, sort_keys=True))
                if tool_calls >= self.MAX_TOOL_HOPS or call_key in self._seen_tool_calls:
                    yield self._force_final_answer(temperature)
                    return
                self._seen_tool_calls.add(call_key)
                result = self._execute_tool(tool_name, params)
                self.stm.add("assistant", content)
                self.stm.add("tool", result, tool_name=tool_name)
                yield from self._call_llm_stream(temperature, tool_calls + 1)
                return
            # False alarm — "ACTION:" appeared but no real tool was named
            # (e.g. "ACTION: 无").  The whole turn is actually the final answer.
            cleaned = self._clean_response(content)
            yield cleaned or "抱歉，暂时未能生成有效回答。请换个问法后重试。"

    # ── helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _clean_response(content: str) -> str:
        """Strip THOUGHT / ACTION / PARAMS scaffold from a final answer.

        Only called when no real tool call was detected.  Also drops the whole
        *leading* scaffold block (a multi-line ``THOUGHT:`` preamble followed
        by a blank line) so wrapped thinking does not leak to the user.
        """
        lines = content.split("\n")

        # Drop a leading scaffold block: from the first scaffold line up to
        # the first blank line that terminates it.
        i = 0
        while i < len(lines) and not lines[i].strip():
            i += 1
        if i < len(lines) and _is_scaffold_line(lines[i]):
            j = i
            while j < len(lines) and lines[j].strip():
                j += 1
            lines = lines[j:]

        cleaned = [l for l in lines if not _is_scaffold_line(l)]
        return "\n".join(cleaned).strip()

    def _parse_tool_call(self, content: str):
        """Parse THOUGHT / ACTION / PARAMS from LLM output.

        Only returns a tool call when the ACTION names a tool we actually
        know.  The model sometimes writes ``ACTION: 无`` (or ``none``) to
        mean "no tool needed" — those must NOT be treated as a call.
        """
        action_match = re.search(r"ACTION:\s*(\S+)", content)
        if not action_match:
            return None
        tool_name = action_match.group(1).strip()
        if tool_name not in self._TOOL_PARAM_MAP:
            return None
        params = {}
        params_match = re.search(r"PARAMS:\s*(\{.*?\})", content, re.DOTALL)
        if params_match:
            try:
                params = json.loads(params_match.group(1))
            except json.JSONDecodeError:
                pass
        return tool_name, params

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
