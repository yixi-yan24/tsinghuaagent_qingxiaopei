import json
from .prompts import PLANNER_PROMPT
from .memory import LongTermMemory
from .course_graph import parse_courses_from_table, topological_sort, format_plan
from .multi_agent import MultiAgentSystem
from .llm_client import chat_completion


class CoursePlanner:
    """Dual-mode planner: algorithmic (topological sort) + LLM-enhanced."""

    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com/v1"):
        self.api_key = api_key
        self.base_url = base_url
        self.multi_agent = MultiAgentSystem(api_key, base_url)

    def generate_plan(
        self,
        major: str,
        grade: str,
        program_name: str,
        ltm: LongTermMemory
    ) -> str:
        prog = ltm.find_program(program_name)
        if not prog:
            return f"未找到培养方案: {program_name}"

        # ========== 创新1: 算法拓扑排序规划 ==========
        courses = parse_courses_from_table(prog.raw_text)
        if courses:
            algo_plan = topological_sort(courses)
            algo_text = format_plan(algo_plan, grade)
        else:
            algo_text = "未能从培养方案中解析出课程表。"

        # ========== LLM 增强规划 ==========
        prompt = PLANNER_PROMPT.format(
            major=major,
            grade=grade,
            program_name=program_name,
            program_text=prog.raw_text[:12000]
        )

        augment = (
            f"\n\n此外，以下是通过课程先修关系拓扑排序算法自动生成的参考计划，"
            f"请结合你的专业知识对其进行改进和补充：\n{algo_text}"
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "你是清华大学课程规划专家。以下有两个规划结果：\n"
                    "1. 算法生成的拓扑排序计划（基于课程先修关系DAG）\n"
                    "2. 你自己需要生成的优化计划\n"
                    "请结合算法计划的学期划分和先修顺序，补充课程解读和学习建议，"
                    "使最终计划既科学又易于理解。"
                )
            },
            {"role": "user", "content": prompt + augment}
        ]

        try:
            llm_response = self._call_llm(messages)

            # ========== 创新2: Multi-Agent 审核 ==========
            verification = self.multi_agent.verify_plan(llm_response, prog.raw_text[:8000])

            # Build final output
            parts = [
                f"## 📋 {program_name} — 修读计划\n",
                llm_response,
                "\n---\n",
            ]

            # Append verification result
            if "审核通过" in verification:
                parts.append("✅ **Multi-Agent 审核**：计划已通过验证。")
            else:
                parts.append(f"🔍 **Multi-Agent 审核意见**：\n{verification}")
                parts.append("\n*以上意见供参考，请结合实际情况调整。*")

            # Append metadata
            if prog.contact:
                parts.append(f"\n\n📞 咨询电话：{prog.contact}")
            if prog.degree:
                parts.append(f"\n🎓 授予学位：{prog.degree}")
            if prog.duration:
                parts.append(f"\n⏱ 学制：{prog.duration}")

            return "\n".join(parts)

        except Exception as e:
            return f"生成计划时出错: {e}\n\n算法生成的参考计划：\n{algo_text}"

    def _call_llm(self, messages: list[dict]) -> str:
        return chat_completion(
            self.api_key, self.base_url, messages,
            temperature=0.3, max_tokens=4096, timeout=90, retries=1,
        )
