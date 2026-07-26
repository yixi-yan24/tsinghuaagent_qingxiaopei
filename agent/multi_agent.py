import json, re
from .memory import ShortTermMemory, LongTermMemory
from .tools import Tools
from .course_graph import parse_courses_from_table, topological_sort, format_plan, build_prerequisite_graph
from .llm_client import chat_completion


class SpecialistAgent:
    """Base class for specialist sub-agents."""

    def __init__(self, name: str, system_prompt: str, api_key: str, base_url: str):
        self.name = name
        self.system_prompt = system_prompt
        self.api_key = api_key
        self.base_url = base_url
        self.memory = ShortTermMemory(max_turns=5)
        self.memory.add("system", system_prompt)

    def think(self, task: str) -> str:
        self.memory.add("user", task)
        response = self._call_llm()
        self.memory.add("assistant", response)
        return response

    def _call_llm(self, temperature: float = 0.3) -> str:
        # Preserve tool_name metadata in tool messages (maintenance safety).
        messages = []
        for m in self.memory.get_all():
            content = m.content
            if m.role == "tool" and m.tool_name:
                content = f"[工具 {m.tool_name}] 结果: {content}"
            messages.append({"role": m.role, "content": content})
        return chat_completion(
            self.api_key, self.base_url, messages,
            temperature=temperature, max_tokens=2048, timeout=60, retries=1,
        )


# === Specialized Agent Definitions ===

PROFILE_AGENT_PROMPT = """你是清华大学培养方案学生档案分析师。
你的任务是从学生对话中提取关键信息，以 JSON 格式输出。

提取字段：
- major: 专业（如 "计算机科学与技术"）
- grade: 年级（如 "大一"、"大二"）
- interests: 兴趣方向（如 "人工智能"、"经济金融"）
- completed_courses: 已修相关课程列表（如果有）
- concerns: 学生提出的顾虑（如果有）

只输出 JSON，不要其他内容。
"""

SEARCH_AGENT_PROMPT = """你是清华大学培养方案搜索专家。
你会收到学生档案和可用的培养方案列表。
你的任务是根据学生的专业和兴趣，推荐最适合的培养方案和课程方向。

要求：
1. 优先匹配学生所在专业的培养方案
2. 考虑学生的兴趣方向，推荐相关选修课程
3. 注意课程之间的先修关系

输出推荐的培养方案名称及简短推荐理由。
"""

VERIFIER_AGENT_PROMPT = """你是清华大学课程规划审核专家。
你会收到一份修读计划，请检查以下方面的问题：

1. 先修课程是否安排正确（先修课应在前置学期）
2. 每学期学分是否合理（一般不超过 25 学分）
3. 是否遗漏了必修课程
4. 总学分是否达到培养方案要求

如果发现问题，请指出具体问题；如果无误，回复"计划审核通过"。
"""


class MultiAgentSystem:
    """Multi-agent orchestrator with specialized sub-agents."""

    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com/v1"):
        self.api_key = api_key
        self.base_url = base_url

        self.profile_agent = SpecialistAgent("profile", PROFILE_AGENT_PROMPT, api_key, base_url)
        self.search_agent = SpecialistAgent("search", SEARCH_AGENT_PROMPT, api_key, base_url)
        self.verifier_agent = SpecialistAgent("verifier", VERIFIER_AGENT_PROMPT, api_key, base_url)

    def analyze_profile(self, conversation_history: list[dict]) -> dict:
        """Extract structured student profile from conversation."""
        history_text = json.dumps(conversation_history[-4:], ensure_ascii=False)
        result = self.profile_agent.think(f"从以下对话中提取学生档案：\n{history_text}")
        try:
            # Extract JSON from response
            json_match = re.search(r"\{.*\}", result, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
        return {"major": "", "grade": "", "interests": ""}

    def search_recommendations(self, profile: dict, ltm: LongTermMemory) -> str:
        """Search and recommend programs based on profile.

        Uses semantic search to pre-filter the full program list so every one of
        the programs has a chance to be recommended, not just the first 20.
        """
        # ── pre-filter: surface the most relevant programs ────────────────
        candidates = self._pre_filter(profile, ltm, top_k=20)

        programs_overview = "\n".join(
            f"- {m.name} ({m.department})"
            for m in candidates
        )
        task = (
            f"学生档案：\n"
            f"专业：{profile.get('major', '未知')}\n"
            f"年级：{profile.get('grade', '未知')}\n"
            f"兴趣：{profile.get('interests', '未知')}\n\n"
            f"可选培养方案列表：\n{programs_overview}\n\n"
            f"请推荐最适合的培养方案和课程方向。"
        )
        return self.search_agent.think(task)

    @staticmethod
    def _pre_filter(profile: dict, ltm: LongTermMemory, top_k: int = 20):
        """Return the *top_k* programs most relevant to the student profile.

        Tries semantic search first; falls back to keyword scoring.
        """
        all_programs = ltm.programs
        if len(all_programs) <= top_k:
            return all_programs

        interests = profile.get("interests", "")
        major = profile.get("major", "")
        query = f"{major} {interests}".strip()

        # 1) Semantic search (embedding-based) — covers all programs.
        if query:
            try:
                from .embedding import semantic_search as _semantic_search
                results = _semantic_search(query, all_programs, top_k=top_k)
                return [m for m, _ in results]
            except Exception:
                pass

        # 2) Keyword fallback — still scans ALL programs.
        if query:
            from .data_loader import search_programs as _search_programs
            kw_results = _search_programs(query, all_programs)
            if kw_results:
                return kw_results[:top_k]

        # 3) Last resort: return first top_k (better than silently hiding half).
        return all_programs[:top_k]

    def verify_plan(self, plan_text: str, program_text: str) -> str:
        """Verify a generated plan for correctness."""
        task = (
            f"【培养方案】\n{program_text[:1500]}\n\n"
            f"【修读计划】\n{plan_text}\n\n"
            f"请审核此计划的合理性。"
        )
        return self.verifier_agent.think(task)
