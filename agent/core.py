import json, re, asyncio
from collections.abc import Generator, AsyncGenerator
from .memory import ShortTermMemory, LongTermMemory
from .llm_client import (
    chat_completion, chat_completion_stream,
    chat_completion_with_tools, chat_completion_stream_with_tools,
    achat_completion, achat_completion_stream,
    achat_completion_with_tools, achat_completion_stream_with_tools,
)
from .tools import Tools
from .planner import CoursePlanner
from .prompts import SYSTEM_PROMPT
from .data_loader import load_programs

# ── scaffold markers the LLM emits for prompt-based tool calling ────────
_SCAFFOLD_PREFIXES = (
    "THOUGHT:", "ACTION:", "PARAMS:",
    "THOUGHT：", "ACTION：", "PARAMS：",   # 全角冒号变体
    "thought:", "action:", "params:",     # 小写变体
)
# 独立一行、无冒号的标记（畸形输出残留，如 "PARAMS" 后换行再写 JSON）。
_SCAFFOLD_BARE = ("无", "none", "None", "不需要", "不需要工具", "无工具",
                  "THOUGHT", "ACTION", "PARAMS")

# ACTION / PARAMS 正则：兼容半角/全角冒号、大小写、前后空格
_ACTION_RE = re.compile(r"ACTION\s*[：:]\s*(\S+)", re.IGNORECASE)
_PARAMS_RE = re.compile(r"PARAMS\s*[：:]\s*", re.IGNORECASE)

# Claude/DeepSeek 风格的工具调用标记：模型偶尔在正文里模仿工具调用格式，
# 这些标记必须从用户可见输出中隐藏。实测会出现的畸形变体包括：
#   <tool_calls>           标准半角
#   ＜tool_calls＞         全角尖括号（U+FF1C/U+FF1E）
#   <｜tool_calls｜>       全角竖线（U+FF5C）充当分隔符
#   <｜DSML｜tool_calls｜> 带 DSML(DeepSeek Markup Language) 前缀
# 因此匹配时：尖括号后跳过任意非字母噪声字符 + 可选 DSML 前缀，再匹配标签名。
_TOOL_XML_RE = re.compile(
    r"^\s*[<＜]/?[^\w]*(?:DSML[^\w]*)?"
    r"(tool_calls|invoke|parameter|function|tool_call|"
    r"antml:invoke|antml:parameter|antml:function)\b",
    re.IGNORECASE,
)

# 无前缀"伪思考句"模式：模型在工具调用后偶尔只输出思考而不给最终回答，
# 且没有 THOUGHT: 前缀（如 "现在我可以使用xxx工具为用户提供推荐……"）。
# 这类句子短、单行、以意图/连接词开头，不是合格的最终回答。
_PSEUDO_THOUGHT_RE = re.compile(
    r"^(让我|我先|现在我可以|接下来我|我打算|我需要先|我需要|让我先|我可以使用|"
    r"好的，让我|好的让我|我已经获得|我已经有|我现在可以|那么我可以|"
    r"接下来我可以|我再|我现在需要|我还需要|用户是|我应该|那么我应该|"
    r"接下来应该|我认为|不过我需要|不过让我|同时我可以|同时，我可以|"
    r"首先，让我|首先搜索|首先需要|先让我|我可以先|让我再|首先我|"
    r"由于用户|根据用户|用户没有|由于|那么我|这轮|本轮|这次|这个用户|"
    r"我找到了|我找到|我查到|我已找到|我搜到).{0,120}$"
)


def _is_scaffold_line(line: str) -> bool:
    """True if a line looks like THOUGHT/ACTION/PARAMS scaffold, a bare
    ``无`` meaning "no tool needed", or a Claude-style XML tool-call tag —
    anything that must be hidden from the user."""
    s = line.strip()
    return (
        s.startswith(_SCAFFOLD_PREFIXES)
        or s in _SCAFFOLD_BARE
        or bool(_TOOL_XML_RE.match(s))
    )


def _looks_like_pseudo_thought(text: str) -> bool:
    """True if *text* looks like bare thinking instead of a final answer.

    Single-line: short and starts with an intention verb.  Multi-line: short
    overall and most lines are short thinking sentences (a model that emits
    nothing but "让我…/我需要…" lines has not produced a real answer).
    """
    t = text.strip()
    if not t:
        return False
    if "\n" in t:
        lines = [l.strip() for l in t.split("\n") if l.strip()]
        if not lines or len(t) > 800:
            return False
        thought = sum(
            1 for l in lines
            if len(l) <= 120 and _PSEUDO_THOUGHT_RE.match(l)
        )
        return thought >= max(1, len(lines) * 0.5)
    return len(t) <= 150 and bool(_PSEUDO_THOUGHT_RE.match(t))


def _strip_pseudo_thought_paragraphs(text: str) -> str:
    """Remove thinking paragraphs the model sometimes emits without a
    THOUGHT: prefix (e.g. "让我给用户一个全面的课程推荐和规划建议。").

    Every paragraph is checked independently (not just the leading run), so a
    thinking sentence in the middle of the reply is dropped too.  A paragraph
    counts as thinking when it is short and its first line reads like an
    intention sentence (or the whole single-line paragraph does).
    """
    paras = [p.strip() for p in text.split("\n\n")]
    kept = []
    for p in paras:
        if not p:
            continue
        lines = p.split("\n")
        is_thought = (
            _looks_like_pseudo_thought(p)
            or (len(lines) <= 4 and _looks_like_pseudo_thought(lines[0].strip()))
        )
        if not is_thought:
            kept.append(p)
    return "\n\n".join(kept).strip()


def _extract_json_params(content: str) -> dict:
    """Extract the JSON object after a ``PARAMS:`` marker.

    Uses json.JSONDecoder.raw_decode so nested braces inside the payload
    (e.g. ``PARAMS: {"a": {"b": 1}}``) are handled correctly instead of
    stopping at the first ``}``.
    """
    m = _PARAMS_RE.search(content)
    if not m:
        return {}
    start = m.end()
    brace = content.find("{", start)
    if brace < 0:
        return {}
    try:
        obj, _ = json.JSONDecoder().raw_decode(content[brace:])
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    return {}


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

    # 明确意图关键词：命中即开启思考链。注意"我对人工智能感兴趣"这类表达
    # 也要命中，因此包含"感兴趣/兴趣/想学/帮我"等。
    _INTENT_KEYWORDS = (
        "排课", "规划", "计划", "推荐", "分析", "建议", "比较", "对比",
        "为什么", "如何", "怎么", "方案", "安排", "策略", "课程表",
        "修读", "选课", "培养", "适合", "合适", "转专业", "保研", "深造",
        "研究", "从事", "方向", "发展", "感兴趣", "兴趣", "想学", "想了解",
        "想研究", "希望", "帮我", "想", "了解",
    )

    # 简单寒暄：命中则不开启思考（直接快速回答）。
    _GREETING_KEYWORDS = ("你好", "嗨", "hi", "hello", "谢谢", "再见", "嗯", "在吗", "好的", "哈哈")

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

    @staticmethod
    def _should_think(user_message: str) -> bool:
        """True if the question warrants a thinking chain.

        默认对多数问题开启思考（能看到推理链、回答更扎实）；只有明显简单
        的寒暄/短句关闭，以保证响应速度。
        """
        msg = user_message.strip()
        if not msg:
            return False
        # 简短寒暄 → 不思考
        if len(msg) <= 6 and any(kw in msg for kw in AgentSession._GREETING_KEYWORDS):
            return False
        # 含明确意图 → 思考
        if any(kw in msg for kw in AgentSession._INTENT_KEYWORDS):
            return True
        # 较长消息（>= 8 字符）默认视为有内容的问题 → 思考
        return len(msg) >= 8

    async def process_message(self, user_message: str, temperature: float = 0.3) -> str:
        """Async: process a user message and return the agent response."""
        self.stm.add("user", user_message)
        response = await self._call_llm(
            temperature=temperature,
            enable_thinking=self._should_think(user_message),
        )
        self.stm.add("assistant", response)
        return response

    async def process_message_events(
        self, user_message: str, temperature: float = 0.3
    ) -> AsyncGenerator[dict, None]:
        """Async: process a user message, yielding structured events.

        Event types:
          {"type": "reasoning", "text": str}   — 思考链增量
          {"type": "content", "text": str}     — 最终回答增量
          {"type": "tool", "name": str}        — 工具执行状态
        """
        self.stm.add("user", user_message)
        should_think = self._should_think(user_message)
        full_content: list[str] = []
        full_reasoning: list[str] = []      # 仅最终轮的思考（中间轮已各自保存）
        current_reasoning: list[str] = []
        try:
            async for ev in self._call_llm_stream(
                temperature=temperature, enable_thinking=should_think
            ):
                if ev["type"] == "content":
                    if current_reasoning:
                        full_reasoning = current_reasoning
                        current_reasoning = []
                    full_content.append(ev["text"])
                elif ev["type"] == "reasoning":
                    current_reasoning.append(ev["text"])
                elif ev["type"] == "tool":
                    current_reasoning = []       # 新一轮工具调用，思考重新累积
                yield ev
        except Exception:
            # 最后兜底：任何内部异常都不允许"静默断流"，给出可用的回答。
            if not full_content:
                try:
                    fallback_text = await self._force_final_answer(temperature)
                    if fallback_text:
                        yield {"type": "content", "text": fallback_text}
                        full_content.append(fallback_text)
                except Exception:
                    yield {"type": "content",
                           "text": "抱歉，回答生成过程中出现异常，请稍后重试或换个问法。"}
        self.stm.add(
            "assistant", "".join(full_content),
            reasoning_content="".join(full_reasoning),
        )

    def process_message_stream(
        self, user_message: str, temperature: float = 0.3
    ) -> Generator[str, None, None]:
        """Sync backward-compatible wrapper: yield the final answer once.

        Prefer the async :meth:`process_message_events` in production; this
        helper exists for old CLI/test callers and runs the async flow inside
        ``asyncio.run``.
        """
        async def _run() -> str:
            parts: list[str] = []
            async for ev in self.process_message_events(user_message, temperature):
                if ev["type"] == "content":
                    parts.append(ev["text"])
            return "".join(parts)

        result = asyncio.run(_run())
        yield result

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
            "\n\n你可以通过系统提供的函数调用能力使用以下工具获取信息。",
            "需要时直接调用工具即可，不要在正文里输出工具调用的格式或标签。",
            "获得工具结果后，直接输出给用户的最终回答正文。",
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

    def _build_tools_schema(self) -> list[dict]:
        """Build DeepSeek native function-calling tools from tool descriptions.

        The model receives these as structured ``tools`` so it returns
        ``tool_calls`` (JSON) instead of free-text THOUGHT/ACTION/PARAMS —
        which eliminates thinking-text leakage at the source.
        """
        tools = []
        for t in self.tools.get_tool_descriptions():
            props = {}
            required = []
            for pname, pinfo in t["parameters"].items():
                props[pname] = {
                    "type": pinfo.get("type", "string"),
                    "description": pinfo.get("description", ""),
                }
                required.append(pname)
            tools.append({
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": {
                        "type": "object",
                        "properties": props,
                        "required": required,
                    } if props else {"type": "object", "properties": {}},
                },
            })
        return tools

    def _native_calls_to_api(self, calls: list[dict]) -> list[dict]:
        """Convert parsed native calls to the OpenAI tool_calls wire format."""
        return [{
            "id": c["id"],
            "type": "function",
            "function": {
                "name": c["name"],
                "arguments": json.dumps(c["arguments"], ensure_ascii=False),
            },
        } for c in calls if c["name"]]

    async def _call_llm(self, temperature: float = 0.3, tool_calls: int = 0,
                        enable_thinking: bool = False) -> str:
        """Async: call DeepSeek API and handle tool use via native function calling.

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

        # 思考策略：复杂问题首轮开启；工具执行后的综合轮始终开启。
        use_thinking = (tool_calls > 0) or enable_thinking
        thinking_payload = {"type": "enabled"} if use_thinking else {"type": "disabled"}
        max_tok = 16384 if use_thinking else 4096

        content = ""
        native_calls: list[dict] = []
        reasoning = ""

        # ── 优先原生 function calling（结构化 tool_calls，无思考泄漏）──
        try:
            content, native_calls, reasoning = await achat_completion_with_tools(
                self.api_key, self.base_url, augmented_messages,
                self._build_tools_schema(),
                temperature=temperature, max_tokens=max_tok, timeout=240, retries=1,
                thinking=thinking_payload,
            )
        except Exception:
            # 原生调用失败时回退到无工具调用（老路径），由下方文本解析兜底
            content = await achat_completion(
                self.api_key, self.base_url, augmented_messages,
                temperature=temperature, max_tokens=max_tok, timeout=240, retries=1,
            )

        if native_calls:
            if tool_calls == 0:
                self._seen_tool_calls.clear()
            call_keys = [
                json.dumps({"name": c["name"], "args": c["arguments"]},
                           ensure_ascii=False, sort_keys=True)
                for c in native_calls if c["name"]
            ]
            # Loop is not converging — force a direct answer.
            if tool_calls >= self.MAX_TOOL_HOPS or any(
                k in self._seen_tool_calls for k in call_keys
            ):
                return await self._force_final_answer(temperature)
            for k in call_keys:
                self._seen_tool_calls.add(k)

            self.stm.add("assistant", content,
                         tool_calls=self._native_calls_to_api(native_calls),
                         reasoning_content=reasoning)
            for c in native_calls:
                if not c["name"]:
                    continue
                result = await self._execute_tool_async(c["name"], c["arguments"])
                self.stm.add("tool", result, tool_name=c["name"], tool_call_id=c["id"])
            return await self._call_llm(temperature, tool_calls + 1, enable_thinking)

        # ── fallback：文本解析（模型未使用原生工具时）──────────────
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
                return await self._force_final_answer(temperature)
            self._seen_tool_calls.add(call_key)
            result = await self._execute_tool_async(tool_name, params)
            self.stm.add("assistant", content)
            self.stm.add("tool", result, tool_name=tool_name)
            return await self._call_llm(temperature, tool_calls + 1, enable_thinking)

        # The model emitted an ACTION marker we could not parse (wrong tool
        # name, full-width colon, ...) — it clearly wanted a tool, so force a
        # clean final answer instead of leaking thinking text.
        if _ACTION_RE.search(content):
            return await self._force_final_answer(temperature)

        cleaned = self._clean_response(content)
        cleaned = _strip_pseudo_thought_paragraphs(cleaned)
        if cleaned and not _looks_like_pseudo_thought(cleaned):
            return cleaned
        return await self._force_final_answer(temperature)

    @staticmethod
    def _extract_answer_block(content: str) -> str:
        """Pull the model's answer out of a ```answer ... ``` code block.

        In forced-answer mode the model is told to wrap its reply in such a
        block, which makes scaffold/thinking material easy to discard.  The
        closing fence may or may not be followed by a newline, so the regex
        does not require one.
        """
        match = re.search(r"```answer[^\n]*\n(.*?)```", content, re.DOTALL)
        if match:
            return match.group(1).strip()
        match = re.search(r"```[^\n]*\n(.*?)```", content, re.DOTALL)
        return match.group(1).strip() if match else ""

    async def _force_final_answer(self, temperature: float = 0.3) -> str:
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

        sys_content = SYSTEM_PROMPT + (
            "\n\n请直接回答用户的问题，给出完整、具体的回答。"
            "必须把最终回答完整地放在 ```answer 和 ``` 代码块之间，"
            "代码块之外不要输出任何其他内容。"
            "不要输出思考过程，不要出现 THOUGHT / ACTION / PARAMS 等标记，"
            "也不要再尝试搜索或调用工具。"
        )
        if tool_results:
            sys_content += (
                "\n\n以下是本次查询过程中收集到的工具结果，请依据这些结果回答；"
                "如果结果中不包含用户需要的信息，请如实告知资料未被收录。"
            )
        else:
            sys_content += "\n如果资料中不包含用户需要的信息，请如实告知资料未被收录。"

        final_messages = [{"role": "system", "content": sys_content}]
        if user_question:
            final_messages.append({"role": "user", "content": user_question})
        for r in tool_results:
            final_messages.append({"role": "user", "content": r})

        content = await achat_completion(
            self.api_key, self.base_url, final_messages,
            temperature=temperature, max_tokens=4096, timeout=180, retries=1,
        )
        result = self._extract_answer_block(content)
        if not result:
            # Model failed to use the fence — one strict retry.
            final_messages.append({"role": "user", "content": "请把回答放在 ```answer 和 ``` 之间，直接输出回答正文。"})
            content = await achat_completion(
                self.api_key, self.base_url, final_messages,
                temperature=temperature, max_tokens=4096, timeout=180, retries=1,
            )
            result = self._extract_answer_block(content)
        if not result:
            # Still no fence — strip scaffold and leading thinking paragraphs.
            result = _strip_pseudo_thought_paragraphs(self._clean_response(content))
        if not result:
            return "抱歉，暂时未能生成有效回答。您可以换个问法，或缩小问题范围后重试。"
        # Defensive: drop any leading thinking paragraph even inside the fence.
        return _strip_pseudo_thought_paragraphs(result)

    async def _call_llm_stream(
        self, temperature: float = 0.3, tool_calls: int = 0,
        enable_thinking: bool = False,
    ) -> AsyncGenerator[dict, None]:
        """Async streaming variant of _call_llm, yielding structured events.

        Event types: {"type": "reasoning"|"content"|"tool", ...}.  Uses native
        function calling with thinking mode; text parsing is kept as fallback.
        """
        messages = self.stm.to_llm_format()
        tool_block = self._get_tool_block()
        if tool_calls > 0:
            # 工具已执行过：鼓励模型直接给出最终回答，防止只输出思考过渡句。
            tool_block += (
                "\n\n【提示】如果你已经通过工具获得了回答用户问题所需的信息，"
                "请直接输出最终回答正文；不要重复调用相同工具，"
                "也不要只输出思考过程或没有实质内容的过渡句。"
            )

        augmented_messages = []
        for msg in messages:
            if msg["role"] == "system":
                augmented_messages.append({
                    "role": "system",
                    "content": msg["content"] + "\n" + tool_block
                })
            else:
                augmented_messages.append(msg)

        # 思考策略：复杂问题首轮开启；工具执行后的综合轮始终开启。
        use_thinking = (tool_calls > 0) or enable_thinking
        thinking_payload = {"type": "enabled"} if use_thinking else {"type": "disabled"}
        max_tok = 16384 if use_thinking else 4096

        # ── native function-calling buffers ─────────────────────────────
        native_calls: dict[int, dict] = {}
        is_native_tool_call = False
        reasoning_buf: list[str] = []   # 累积思考内容，多轮需回传
        reasoning_pending = ""          # 行级过滤思考中的 XML 标签

        # Text-fallback streaming state (used when the model emits plain text
        # instead of native tool_calls).  The window is tiny — native tool
        # calls arrive as structured tool_call events, so content can start
        # streaming almost immediately (only a bare thinking first line is
        # buffered a bit longer to avoid leaking it).
        TOOL_DETECT_WINDOW = 30  # chars before starting to stream content

        buffered: list[str] = []
        is_tool_call = False
        yielded = False
        pending_line = ""
        skip_until_blank = False

        def emit(tokens: list[str]) -> list[dict]:
            """Return content events line by line, dropping scaffold material."""
            nonlocal pending_line, skip_until_blank
            out: list[dict] = []
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
                    if tool_calls > 0 and _looks_like_pseudo_thought(s):
                        continue
                    out.append({"type": "content", "text": line + "\n"})
            return out

        def flush_tail() -> list[dict]:
            nonlocal pending_line
            if not pending_line:
                return []
            line, pending_line = pending_line, ""
            if not _is_scaffold_line(line) and not (
                tool_calls > 0 and _looks_like_pseudo_thought(line.strip())
            ):
                return [{"type": "content", "text": line}]
            return []

        async def _content_events(gen) -> AsyncGenerator[dict, None]:
            """Wrap a plain async token stream into content events."""
            async for token in gen:
                yield {"type": "content", "text": token}

        async def _iter_events() -> AsyncGenerator[dict, None]:
            """Stream native function-calling events, falling back to a plain
            text stream if the tools request fails mid-stream (network drop,
            timeout, HTTP error, …).  Emits a ``{"type": "fallback"}`` marker
            so the caller abandons any half-collected tool call and streams
            the fallback answer instead of silently dropping it.
            """
            try:
                async for ev in achat_completion_stream_with_tools(
                    self.api_key, self.base_url, augmented_messages,
                    self._build_tools_schema(),
                    temperature=temperature, max_tokens=max_tok, timeout=240,
                    thinking=thinking_payload,
                ):
                    yield ev
            except Exception:
                yield {"type": "fallback"}
                async for ev in _content_events(achat_completion_stream(
                    self.api_key, self.base_url, augmented_messages,
                    temperature=temperature, max_tokens=max_tok, timeout=240,
                )):
                    yield ev

        async for event in _iter_events():
            if event["type"] == "fallback":
                # 原生工具流式中断：放弃未完成的工具调用，直接输出 fallback 回答。
                is_native_tool_call = False
                native_calls.clear()
                buffered.clear()
                reasoning_buf.clear()
                reasoning_pending = ""
                continue

            if event["type"] == "reasoning":
                # Thinking chain — surface it to the user AND keep it for
                # multi-turn replay (DeepSeek requires echoing it back).  Filter
                # out any Claude-style XML tool-call tags the model may emit
                # inside its thinking.
                reasoning_buf.append(event["text"])
                reasoning_pending += event["text"]
                while "\n" in reasoning_pending:
                    line, reasoning_pending = reasoning_pending.split("\n", 1)
                    if not _is_scaffold_line(line):
                        yield {"type": "reasoning", "text": line + "\n"}
                continue

            if event["type"] == "tool_call":
                # Native tool call — concatenate incremental argument fragments.
                is_native_tool_call = True
                idx = event["index"]
                entry = native_calls.setdefault(idx, {"id": "", "name": "", "arguments": ""})
                if event["id"]:
                    entry["id"] = event["id"]
                if event["name"]:
                    entry["name"] = event["name"]
                entry["arguments"] += event["arguments"]
                # Any content buffered before the tool call was thinking —
                # discard it so nothing leaks on the text-fallback path.
                buffered.clear()
                continue

            token = event["text"]
            if is_native_tool_call:
                # Any content emitted alongside a tool call is model thinking
                # — discard it (the structured tool_calls are authoritative).
                continue

            if is_tool_call:
                buffered.append(token)
                continue
            if not yielded:
                buffered.append(token)
                joined = "".join(buffered)
                if "ACTION:" in joined or "ACTION：" in joined:
                    # Text-based tool call detected — keep buffering silently.
                    is_tool_call = True
                elif len(joined) >= TOOL_DETECT_WINDOW:
                    first_line = joined.split("\n", 1)[0].strip()
                    if _looks_like_pseudo_thought(first_line) or "PARAMS" in joined:
                        pass
                    else:
                        # Looks like a normal response — stream it (scaffold-filtered).
                        yielded = True
                        for _d in emit(buffered):
                            yield _d
                        buffered.clear()
            else:
                # Already streaming a normal answer — keep going even if a
                # stray "ACTION:" appears; it is just part of the text.
                for _d in emit([token]):
                    yield _d

        # Flush any trailing reasoning line (not ending with a newline).
        if reasoning_pending.strip() and not _is_scaffold_line(reasoning_pending):
            yield {"type": "reasoning", "text": reasoning_pending}

        # ── native tool-call handling ───────────────────────────────────
        if is_native_tool_call:
            calls: list[dict] = []
            for idx in sorted(native_calls):
                entry = native_calls[idx]
                if not entry["name"]:
                    continue
                try:
                    args = json.loads(entry["arguments"]) if entry["arguments"] else {}
                except json.JSONDecodeError:
                    args = {}
                calls.append({
                    "id": entry["id"],
                    "name": entry["name"],
                    "arguments": args if isinstance(args, dict) else {},
                })
            if calls:
                if tool_calls == 0:
                    self._seen_tool_calls.clear()
                call_keys = [
                    json.dumps({"name": c["name"], "args": c["arguments"]},
                               ensure_ascii=False, sort_keys=True)
                    for c in calls
                ]
                if tool_calls >= self.MAX_TOOL_HOPS or any(
                    k in self._seen_tool_calls for k in call_keys
                ):
                    yield {"type": "content", "text": await self._force_final_answer(temperature)}
                    return
                for k in call_keys:
                    self._seen_tool_calls.add(k)
                self.stm.add("assistant", "",
                             tool_calls=self._native_calls_to_api(calls),
                             reasoning_content="".join(reasoning_buf))
                for c in calls:
                    yield {"type": "tool", "name": c["name"]}
                    result = await self._execute_tool_async(c["name"], c["arguments"])
                    self.stm.add("tool", result, tool_name=c["name"], tool_call_id=c["id"])
                async for ev in self._call_llm_stream(temperature, tool_calls + 1, enable_thinking):
                    yield ev
                return
            # tool_call deltas arrived but no complete call — force a clean answer.
            yield {"type": "content", "text": await self._force_final_answer(temperature)}
            return

        # ── text-fallback handling ──────────────────────────────────────
        if is_tool_call:
            content = "".join(buffered)
        elif yielded:
            content = "".join(buffered)
            for _d in flush_tail():
                yield _d
        else:
            # Response shorter than detection window — evaluate as a whole.
            content = "".join(buffered)

        # Check if the LLM wants to use a tool
        if is_tool_call:
            tool_result = self._parse_tool_call(content)
            if tool_result:
                tool_name, params = tool_result
                if tool_calls == 0:
                    self._seen_tool_calls.clear()
                call_key = (tool_name, json.dumps(params, ensure_ascii=False, sort_keys=True))
                if tool_calls >= self.MAX_TOOL_HOPS or call_key in self._seen_tool_calls:
                    yield {"type": "content", "text": await self._force_final_answer(temperature)}
                    return
                self._seen_tool_calls.add(call_key)
                yield {"type": "tool", "name": tool_name}
                result = await self._execute_tool_async(tool_name, params)
                self.stm.add("assistant", content)
                self.stm.add("tool", result, tool_name=tool_name)
                async for ev in self._call_llm_stream(temperature, tool_calls + 1, enable_thinking):
                    yield ev
                return
            # False alarm — an ACTION marker appeared but no real tool was
            # named (wrong name, full-width colon, "ACTION: 无", ...).  The
            # buffer may contain bare thinking text, so force a clean answer.
            yield {"type": "content", "text": await self._force_final_answer(temperature)}
            return

        # ── no tool call ────────────────────────────────────────────────
        if not yielded:
            # Whole response fits in the window: clean it, and if the result
            # is just a bare thinking sentence, force a real answer instead.
            cleaned = _strip_pseudo_thought_paragraphs(self._clean_response(content))
            if cleaned and not _looks_like_pseudo_thought(cleaned):
                yield {"type": "content", "text": cleaned}
            else:
                yield {"type": "content", "text": await self._force_final_answer(temperature)}
            return
        # yielded branch: content was already streamed out above (incl. tail).

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
        action_match = _ACTION_RE.search(content)
        if not action_match:
            return None
        tool_name = action_match.group(1).strip()
        if tool_name not in self._TOOL_PARAM_MAP:
            return None
        return tool_name, _extract_json_params(content)

    def _safe_execute_tool(self, name: str, args: dict) -> str:
        """Execute a tool, returning an error string instead of raising.

        Keeps the assistant(tool_calls) → tool message pairing intact even if
        the tool itself crashes, so the next API call never sees a dangling
        tool_calls entry (which would be rejected with HTTP 400).
        """
        try:
            return self._execute_tool(name, args)
        except Exception as e:
            return f"[工具执行出错] {type(e).__name__}: {e}"

    async def _execute_tool_async(self, name: str, args: dict,
                                  timeout: float = 90) -> str:
        """Run a tool in a worker thread with a hard timeout.

        A slow tool (e.g. multi_agent_search with several nested LLM calls)
        must never stall the whole turn indefinitely — on timeout we return an
        error string so the conversation keeps flowing.
        """
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(self._safe_execute_tool, name, args),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            return f"[工具执行超时] {name} 超过 {int(timeout)}s 未完成，已跳过。"

    def _execute_tool(self, tool_name: str, params: dict) -> str:
        """Dispatch *tool_name* to the matching method on self.tools."""
        tools = self.tools
        # Resolve the method once then call with only the parameters it expects.
        method = getattr(tools, tool_name, None)
        if method is None:
            return f"未知工具: {tool_name}"

        # Normalise parameter values to strings.  The model may send numbers
        # (e.g. "gpa": 3.6) or arrays (e.g. "completed_courses": [...]) in
        # PARAMS — tools expect plain strings, so coerce them here.
        param_names = self._TOOL_PARAM_MAP.get(tool_name, [])
        kwargs = {}
        for p in param_names:
            v = params.get(p, "")
            if isinstance(v, list):
                v = ",".join(str(x) for x in v)
            elif isinstance(v, dict):
                v = json.dumps(v, ensure_ascii=False)
            elif v is None:
                v = ""
            else:
                v = str(v)
            kwargs[p] = v
        try:
            return method(**kwargs)
        except TypeError:
            # Fallback for no-arg tools (e.g. list_programs).
            return method()
