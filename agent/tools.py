from typing import Optional
from .course_catalog import CourseCatalog, CourseRecord
from .scheduler import generate_schedule
from .memory import LongTermMemory


class Tools:
    """Tools available to the agent."""

    def __init__(self, long_term_memory: LongTermMemory, api_key: str = "", course_catalog: Optional[CourseCatalog] = None):
        self.ltm = long_term_memory
        self._api_key = api_key
        self.course_catalog = course_catalog or CourseCatalog.load()
        # Lazily-initialised caches.
        self._multi_agent_system = None

    def list_programs(self, group_by_department: bool = True) -> str:
        """列出所有可用的本科培养方案（按院系分组）。"""
        names = self.ltm.list_all()
        if not group_by_department:
            return "清华大学2025级本科培养方案列表：\n" + "\n".join(f"  {i+1}. {n}" for i, n in enumerate(names))

        # Group by department for compact, scannable output.
        from collections import defaultdict
        grouped: dict[str, list[str]] = defaultdict(list)
        for m in self.ltm.programs:
            dept = m.department or "其他院系"
            grouped[dept].append(m.name)

        lines = ["清华大学2025级本科培养方案列表（按院系分组）："]
        for dept, programs in sorted(grouped.items()):
            lines.append(f"\n【{dept}】")
            for name in programs:
                lines.append(f"  - {name}")
        lines.append(f"\n共 {len(names)} 个培养方案。")
        return "\n".join(lines)

    def search_programs(self, keyword: str) -> str:
        """搜索培养方案。"""
        if not keyword.strip():
            return "请提供专业名称、院系或方向关键词后再搜索。"
        results = self.ltm.search(keyword)
        if not results:
            return f"未找到与 '{keyword}' 相关的培养方案。"
        lines = [f"找到 {len(results)} 个相关培养方案："]
        for m in results:
            lines.append(f"\n【{m.name}】({m.department})")
            lines.append(f"  学分要求：{m.total_credits or '见培养方案'}")
            if m.degree:
                lines.append(f"  授予学位：{m.degree}")
            if m.duration:
                lines.append(f"  学制：{m.duration}")
            if m.contact:
                lines.append(f"  咨询电话：{m.contact}")
        return "\n".join(lines)

    def search_courses(self, keyword: str) -> str:
        """搜索已收录的课程，可按课程号、名称、院系或课程内容关键词查询。"""
        if not keyword.strip():
            return "请提供课程号、课程名称或主题关键词后再搜索。"
        courses = self.course_catalog.search(keyword)
        if not courses:
            return f'未在已收录的课程资料中找到与"{keyword}"相关的课程。'
        lines = [f"找到 {len(courses)} 门相关课程："]
        for course in courses:
            programs = "、".join(
                str(program.get("program", "")).replace("专业辅修培养方案", "").replace("专业培养方案", "")
                for program in course.minor_programs[:3]
            )
            lines.append(
                f"\n【{course.name}】课程号：{course.id}｜{course.department}｜"
                f"{course.credits if course.credits is not None else '未知'} 学分"
            )
            if programs:
                lines.append(f"  关联培养方案：{programs}")
            if course.prerequisites:
                lines.append(f"  先修要求：{course.prerequisites[:200]}")
            if course.description:
                lines.append(f"  内容摘要：{course.description[:240]}")
        return "\n".join(lines)

    def get_course_detail(self, identifier: str) -> str:
        """获取一门已收录课程的课程内容、先修、考核与教材信息。"""
        course = self.course_catalog.find(identifier)
        if not course:
            matches = self.course_catalog.search(identifier, limit=2) if identifier.strip() else []
            if matches:
                return "课程名称不够明确，请从以下候选中指定一门：" + "、".join(
                    f"{item.name}（{item.id}）" for item in matches
                )
            return f"未找到课程：{identifier}"
        return self._format_course_detail(course)

    def list_program_courses(self, program_name: str) -> str:
        """列出某培养方案中已收录详细资料的课程。"""
        program = self.ltm.find_program(program_name)
        if not program:
            return f"未找到培养方案: {program_name}"
        courses = self.course_catalog.for_program(program.name)
        if not courses:
            return f"{program.name} 暂无可用的详细课程资料。"
        lines = [f"{program.name} 已收录详细资料的课程（共 {len(courses)} 门）："]
        for course in courses:
            lines.append(
                f"- {course.name}（{course.id}，{course.credits if course.credits is not None else '未知'} 学分，{course.department}）"
            )
        return "\n".join(lines)

    @staticmethod
    def _format_course_detail(course: CourseRecord) -> str:
        programs = "、".join(
            str(program.get("program", "")).replace("专业辅修培养方案", "").replace("专业培养方案", "")
            for program in course.minor_programs
        ) or "未标注"
        parts = [
            f"【{course.name}】",
            f"课程号：{course.id}",
            f"开课单位：{course.department or '未标注'}",
            f"学分：{course.credits if course.credits is not None else '未标注'}",
            f"总学时：{course.total_hours if course.total_hours is not None else '未标注'}",
            f"关联培养方案：{programs}",
        ]
        fields = [
            ("先修要求", course.prerequisites),
            ("课程内容", course.description),
            ("教学目标", course.objectives),
            ("预期学习成效", course.expected_outcomes),
            ("考核方式", course.assessment_method),
            ("成绩构成", course.grade_breakdown),
            ("教材及参考书", course.textbooks),
            ("课程负责人", course.instructor),
        ]
        for label, value in fields:
            if value:
                parts.append(f"\n{label}：{value[:1200]}")
        return "\n".join(parts)

    def get_program_detail(self, name: str) -> str:
        """获取某个培养方案的详细信息。"""
        prog = self.ltm.find_program(name)
        if not prog:
            return f"未找到培养方案: {name}"
        parts = [
            f"【{prog.name}】",
            f"开设院系：{prog.department}",
            f"学分要求：{prog.total_credits or '见培养方案'}",
        ]
        if prog.degree:
            parts.append(f"授予学位：{prog.degree}")
        if prog.duration:
            parts.append(f"学制：{prog.duration}")
        if prog.prerequisites:
            parts.append(f"先修课程：{prog.prerequisites}")
        if prog.major_restrictions:
            parts.append(f"招生说明：{prog.major_restrictions}")
        if prog.contact:
            parts.append(f"咨询电话：{prog.contact}")
        parts.append(f"\n--- 培养方案详情 ---\n{prog.raw_text[:3000]}")
        return "\n".join(parts)

    def check_requirements(self, major: str, program_name: str) -> str:
        """检查专业培养方案的基本要求。"""
        prog = self.ltm.find_program(program_name)
        if not prog:
            return f"未找到培养方案: {program_name}"
        return (
            f"【培养方案要求】专业：{major} → 培养方案：{prog.name}\n"
            f"学分要求：{prog.total_credits or '见培养方案'}\n"
            f"学位：{prog.degree or '见培养方案'}\n"
            f"学制：{prog.duration or '见培养方案'}\n"
            f"说明：{prog.major_restrictions or '见培养方案详情'}"
        )

    def semantic_search(self, query: str) -> str:
        """【语义搜索】使用词嵌入（Word Embedding）进行语义相似度搜索，理解查询意图而非仅匹配关键词。"""
        try:
            from .embedding import semantic_search as _semantic_search
            # Reuse the already-loaded programs from long-term memory — no re-parse.
            results = _semantic_search(query, self.ltm.programs, top_k=5)
            if not results:
                return f"语义搜索未找到与 '{query}' 相关的培养方案。"
            lines = [f"词嵌入语义搜索 '{query}' 的结果（按相关度排序）："]
            for m, score in results:
                lines.append(f"\n【{m.name}】({m.department}) [相似度: {score:.3f}]")
                lines.append(f"  学分：{m.total_credits or '见方案'}")
                lines.append(f"  说明：{m.major_restrictions[:100] or '无'}")
            return "\n".join(lines)
        except ImportError:
            return "语义搜索不可用：请安装 sentence-transformers 以启用词嵌入功能。"
        except Exception as e:
            return f"语义搜索出错: {e}"

    def multi_agent_search(self, major: str, interests: str, grade: str = "") -> str:
        """【Multi-Agent 协同搜索】使用多个专业子 Agent 协同分析学生需求，推荐最适配的培养方案和课程方向。"""
        try:
            from .multi_agent import MultiAgentSystem
            # Cache the system so sub-agents are reused across calls.
            if self._multi_agent_system is None:
                self._multi_agent_system = MultiAgentSystem(api_key=self._api_key)
            mas = self._multi_agent_system
            profile = {"major": major, "grade": grade, "interests": interests}
            return mas.search_recommendations(profile, self.ltm)
        except Exception as e:
            return f"Multi-Agent 搜索出错: {e}"

    def recommend_courses(self, major: str = "", grade: str = "",
                          interests: str = "", semester: str = "") -> str:
        """【课程推荐】根据学生专业、年级、兴趣和学期，推荐适合的课程。

        综合考虑院系匹配、兴趣关键词、年级适配、课程类型等因素打分排序。
        开课学期数据待后续填充，当前默认所有课程均可推荐。
        """
        if not major and not interests:
            return "请提供你的专业或兴趣方向，以便推荐适合的课程。"
        results = self.course_catalog.recommend(
            major=major, grade=grade, interests=interests,
            target_semester=semester, limit=10,
        )
        if not results:
            return f"未找到与你的条件匹配的课程。请尝试调整关键词或专业信息。"
        lines = [f"基于你的专业（{major or '未指定'}）、年级（{grade or '未指定'}）"]
        if interests:
            lines[0] += f"和兴趣（{interests}）"
        lines[0] += f"，推荐以下 {len(results)} 门课程："
        for score, course in results:
            sem_info = f"｜开课：{course.semester}" if course.semester else ""
            type_info = f"｜{course.course_type}" if course.course_type else ""
            lines.append(
                f"\n【{course.name}】课程号：{course.id}｜"
                f"{course.department}｜{course.credits or '?'}学分{sem_info}{type_info}｜推荐度：{score}"
            )
            if course.prerequisites:
                lines.append(f"  先修要求：{course.prerequisites[:150]}")
            if course.description:
                lines.append(f"  内容摘要：{course.description[:200]}")
        return "\n".join(lines)

    def generate_schedule_tool(self, major: str = "", grade: str = "",
                               program_name: str = "",
                               completed_courses: str = "", gpa: str = "",
                               goals: str = "", target_semester: str = "") -> str:
        """【智能排课】根据学生专业、年级、已修课程、培养方案要求、目标（保研/出国等），
        自动生成按学期的推荐课程表。

        综合考虑：先修关系、开课学期、学分上限（25/学期）、保研核心课前置等约束。
        """
        if not major or not program_name:
            return "请提供专业和培养方案名称，以便生成课程表。"
        try:
            gpa_val = float(gpa) if gpa and gpa.strip() else 0.0
        except ValueError:
            gpa_val = 0.0
        return generate_schedule(
            major=major, grade=grade, program_name=program_name,
            completed_courses=completed_courses, gpa=gpa_val,
            goals=goals, target_semester=target_semester,
            catalog=self.course_catalog,
        )

    def get_tool_descriptions(self) -> list[dict]:
        """Return tool descriptions in a format usable by LLM."""
        return [
            {
                "name": "list_programs",
                "description": "列出所有清华大学2025级本科培养方案",
                "parameters": {}
            },
            {
                "name": "search_programs",
                "description": "根据关键词搜索培养方案",
                "parameters": {
                    "keyword": {"type": "string", "description": "搜索关键词，如专业名称、院系"}
                }
            },
            {
                "name": "get_program_detail",
                "description": "获取某个专业培养方案的详细信息",
                "parameters": {
                    "name": {"type": "string", "description": "培养方案/专业名称"}
                }
            },
            {
                "name": "search_courses",
                "description": "按课程号、课程名称、院系或课程主题搜索已收录的课程",
                "parameters": {
                    "keyword": {"type": "string", "description": "例如：机器学习、建筑设计、30000833"}
                }
            },
            {
                "name": "get_course_detail",
                "description": "获取一门课程的内容简介、先修要求、考核方式和教材等详细资料",
                "parameters": {
                    "identifier": {"type": "string", "description": "精确课程号或明确课程名称"}
                }
            },
            {
                "name": "list_program_courses",
                "description": "列出某培养方案中已收录详细课程资料的课程",
                "parameters": {
                    "program_name": {"type": "string", "description": "培养方案/专业名称"}
                }
            },
            {
                "name": "check_requirements",
                "description": "检查某专业培养方案的学分要求、学位授予、学制等基本信息",
                "parameters": {
                    "major": {"type": "string", "description": "学生的专业"},
                    "program_name": {"type": "string", "description": "培养方案名称"}
                }
            },
            {
                "name": "semantic_search",
                "description": "【词嵌入语义搜索】用 AI 理解查询意图进行语义搜索（如搜索'计算机'也能找到'软件工程'、'人工智能'等）",
                "parameters": {
                    "query": {"type": "string", "description": "搜索查询"}
                }
            },
            {
                "name": "multi_agent_search",
                "description": "【Multi-Agent】使用多个AI专家协同分析，推荐最适配的培养方案和课程方向",
                "parameters": {
                    "major": {"type": "string", "description": "学生的专业"},
                    "interests": {"type": "string", "description": "学生的兴趣方向"},
                    "grade": {"type": "string", "description": "年级"}
                }
            },
            {
                "name": "generate_schedule",
                "description": "【智能排课】根据学生专业、年级、已修课程、培养方案、目标（保研/出国等），自动生成按学期排列的推荐课程表，考虑先修关系、开课学期、学分上限",
                "parameters": {
                    "major": {"type": "string", "description": "学生的专业"},
                    "grade": {"type": "string", "description": "年级（大一/大二/大三/大四）"},
                    "program_name": {"type": "string", "description": "培养方案名称"},
                    "completed_courses": {"type": "string", "description": "已修课程，逗号分隔（如：微积分,线性代数,10421263）"},
                    "gpa": {"type": "string", "description": "当前GPA（可选，用于保研评估）"},
                    "goals": {"type": "string", "description": "目标，逗号分隔（如：保研,出国,就业）"},
                    "target_semester": {"type": "string", "description": "开始排课学期（秋/春/夏），默认秋"}
                }
            },
            {
                "name": "recommend_courses",
                "description": "【课程推荐】根据学生专业、年级、兴趣方向推荐适合的课程，综合考虑院系匹配、年级适配等因素",
                "parameters": {
                    "major": {"type": "string", "description": "学生的专业"},
                    "grade": {"type": "string", "description": "年级（大一/大二/大三/大四）"},
                    "interests": {"type": "string", "description": "兴趣方向（如：人工智能、经济学、建筑设计）"},
                    "semester": {"type": "string", "description": "目标学期（秋/春/夏），可选"}
                }
            }
        ]
