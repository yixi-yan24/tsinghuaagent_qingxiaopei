#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""清华大学本科培养方案助手 — 数据查询 CLI（纯标准库，无外部依赖）。

由 TsingXiaoPeiAgent 的 agent/tools.py、course_catalog.py、data_loader.py、
course_graph.py、scheduler.py 改编：去掉了词嵌入语义搜索、Multi-Agent 与
LLM 调用，保留全部本地数据查询、拓扑排序规划与约束排课能力，
适配为 skill 的辅助脚本。

用法（数据位于 skill 目录的 data/ 下）：
    python advisor.py list_programs [--flat]
    python advisor.py search_programs <关键词>
    python advisor.py get_program_detail <培养方案名>
    python advisor.py check_requirements <主修专业> <培养方案名>
    python advisor.py search_courses <关键词>
    python advisor.py get_course_detail <课程号或课程名>
    python advisor.py list_program_courses <培养方案名>
    python advisor.py recommend_courses <主修专业> <年级> <兴趣> [--semester 秋/春/夏]
    python advisor.py plan <主修专业> <年级> <培养方案名>
    python advisor.py schedule <主修专业> <年级> <培养方案名>
                  [--completed 课程号或课程名,逗号分隔] [--gpa 3.5]
                  [--goals 保研,出国] [--start 秋/春/夏]
"""

import json
import os
import re
import sys
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROGRAMS_PATH = os.path.join(BASE_DIR, "data", "programs.json")
COURSES_PATH = os.path.join(BASE_DIR, "data", "courses.json")

# ══════════════════════════════════════════════════════════════════════
# 数据层（改编自 data_loader.py / course_catalog.py）
# ══════════════════════════════════════════════════════════════════════


@dataclass
class TrainingProgram:
    name: str
    department: str
    total_credits: str = ""
    degree: str = ""
    duration: str = ""
    prerequisites: str = ""
    major_restrictions: str = ""
    contact: str = ""
    raw_text: str = ""


def _normalize(value: str) -> str:
    return re.sub(r"[\s（）()\[\]【】,，.。:：;；、·/\\_-]+", "", value).lower()


def load_programs() -> list[TrainingProgram]:
    with open(PROGRAMS_PATH, encoding="utf-8") as f:
        return [TrainingProgram(**item) for item in json.load(f)]


def get_program_by_name(name: str, programs: list[TrainingProgram]) -> Optional[TrainingProgram]:
    """按名称查找培养方案（容忍标点差异与部分名称），名称未命中时按院系匹配。

    存在同名培养方案（如数学系与致理书院均有"数学与应用数学专业本科培养方案"）时
    返回第一个；调用方可用 _same_name_duplicates 提示院系歧义。
    """
    name = name.strip()
    if not name:
        return None
    q = _normalize(name)
    norm_names = [_normalize(m.name) for m in programs]
    for i, n in enumerate(norm_names):
        if q and (q in n or n in q):
            return programs[i]
    # 按空格/顿号分词匹配（如"计算机 培养方案"）
    for m, n in zip(programs, norm_names):
        for kw in re.split(r"[\s、,，]+", q):
            if kw and kw in n:
                return m
    # 名称未命中时按院系匹配（适用于方案名为通用"本科培养方案"的书院）
    for m in programs:
        if q and q in _normalize(m.department or ""):
            return m
    return None


def _same_name_duplicates(prog: TrainingProgram,
                          programs: list[TrainingProgram]) -> list[TrainingProgram]:
    """返回与 prog 同名的其他培养方案（用于提示院系歧义）。"""
    return [m for m in programs
            if m.name == prog.name and m.department != prog.department]


def search_programs(query: str, programs: list[TrainingProgram]) -> list[TrainingProgram]:
    q = query.strip().lower()
    if not q:
        return []
    scored = []
    for m in programs:
        if q in m.name.lower():
            score = 100
        elif q in m.department.lower():
            score = 70
        elif q in m.prerequisites.lower() or q in m.major_restrictions.lower():
            score = 50
        elif q in m.raw_text.lower():
            score = 30
        else:
            continue
        scored.append((score, m))
    scored.sort(key=lambda item: (-item[0], item[1].name))
    return [program for _, program in scored]


@dataclass
class CourseRecord:
    id: str
    name: str
    department: str = ""
    course_type: str = ""
    credits: Optional[float] = None
    total_hours: Optional[int] = None
    prerequisites: str = ""
    description: str = ""
    objectives: str = ""
    expected_outcomes: str = ""
    assessment_method: str = ""
    grade_breakdown: str = ""
    textbooks: str = ""
    instructor: str = ""
    minor_programs: list[dict] = field(default_factory=list)


def load_courses() -> list[CourseRecord]:
    with open(COURSES_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return [CourseRecord(**item) for item in data if isinstance(item, dict)]


class CourseCatalog:
    """可检索的课程目录，带加权关键词评分。"""

    def __init__(self, courses: list[CourseRecord]):
        self.courses = [c for c in courses if c.id and c.name]
        self._by_id = {c.id: c for c in self.courses}

    def find(self, identifier: str) -> Optional[CourseRecord]:
        query = _normalize(identifier)
        if not query:
            return None
        if identifier.strip() in self._by_id:
            return self._by_id[identifier.strip()]
        exact = [c for c in self.courses if _normalize(c.name) == query]
        if len(exact) == 1:
            return exact[0]
        partial = [c for c in self.courses if query in _normalize(c.name)]
        return partial[0] if len(partial) == 1 else None

    def search(self, query: str, limit: int = 5) -> list[CourseRecord]:
        nq = _normalize(query)
        if not nq:
            return []
        scored = []
        for c in self.courses:
            metadata = _normalize(
                f"{c.department} "
                + " ".join(str(p.get("program", "")) for p in c.minor_programs)
            )
            content = _normalize(
                f"{c.description[:2000]} {c.objectives[:1000]} "
                f"{c.expected_outcomes[:1000]} {c.prerequisites}"
            )
            name = _normalize(c.name)
            score = 0
            if c.id == query.strip():
                score = 100
            elif name == nq:
                score = 95
            elif nq in name:
                score = 80
            elif nq in metadata:
                score = 55
            elif nq in content:
                score = 35
            if score:
                scored.append((score, c))
        scored.sort(key=lambda item: (-item[0], item[1].name, item[1].id))
        return [c for _, c in scored[:limit]]

    def for_program(self, program_name: str, department: str = "",
                    limit: int = 40) -> list[CourseRecord]:
        """找出与培养方案关联的课程。

        课程库的关联字段记录的是辅修培养方案名（如"计算机科学与技术专业辅修培养方案"），
        与本科方案名（"计算机科学与技术专业本科培养方案"）按去除"本科/辅修/培养方案"
        后的名称片段互相包含匹配；方案名为通用"本科培养方案"时退化为按院系匹配。
        """
        query = _normalize(program_name.replace("本科培养方案", "")
                           .replace("专业培养方案", "").replace("培养方案", ""))
        dept_q = _normalize(department or "")
        if not query and not dept_q:
            return []
        matches = []
        for c in self.courses:
            programs_norm = " ".join(
                _normalize(str(p.get("program", ""))) for p in c.minor_programs)
            if query and query in programs_norm:
                matches.append((100, c))
            elif dept_q and _normalize(c.department) and (
                    dept_q[:4] in _normalize(c.department)
                    or _normalize(c.department)[:4] in dept_q):
                matches.append((70, c))
        matches.sort(key=lambda x: (-x[0], x[1].name))
        return [c for _, c in matches[:limit]]

    def recommend(self, major: str = "", grade: str = "", interests: str = "",
                  target_semester: str = "", limit: int = 10) -> list[tuple[int, CourseRecord]]:
        """基于学生画像推荐课程（打分排序，改编自 course_catalog.recommend）。

        评分维度（各 0-25）：院系匹配、兴趣关键词命中、年级适配、
        学期适配（课程库开课学期数据未收录，当前取中性分）。
        """
        interest_keywords = [
            kw.strip() for kw in re.split(r"[、,，]", interests) if kw.strip()
        ] if interests else []
        grade_num = {"大一": 1, "大二": 2, "大三": 3, "大四": 4, "大五": 5}.get(grade, 0)

        scored: list[tuple[int, CourseRecord]] = []
        for course in self.courses:
            score = 0
            # 1) 院系匹配（0-25）
            dept_norm = _normalize(course.department or "")
            major_norm = _normalize(major)
            if major_norm and dept_norm:
                if major_norm[:4] in dept_norm or dept_norm[:4] in major_norm:
                    score += 25
                elif any(t in dept_norm for t in major_norm[:4]):
                    score += 15
                else:
                    score += 5
            # 2) 兴趣匹配（0-25）
            if interest_keywords:
                course_text = _normalize(
                    f"{course.name} {course.description[:500]} {course.objectives[:300]}")
                hits = sum(1 for kw in interest_keywords if _normalize(kw) in course_text)
                if hits >= 3:
                    score += 25
                elif hits >= 2:
                    score += 20
                elif hits >= 1:
                    score += 15
            # 3) 年级适配（0-25）：低年级偏好基础课，高年级偏好进阶课
            course_text = _normalize(f"{course.name} {course.description[:300]}")
            foundation_words = ["基础", "概论", "导论", "入门", "初探", "基本", "原理"]
            advanced_words = ["高级", "前沿", "研究", "专题", "研讨", "实践", "设计"]
            foundation_score = sum(1 for w in foundation_words if w in course_text)
            advanced_score = sum(1 for w in advanced_words if w in course_text)
            if grade_num <= 1:
                score += min(25, foundation_score * 8 + 5)
            elif grade_num <= 2:
                score += min(25, foundation_score * 5 + advanced_score * 5 + 5)
            elif grade_num >= 3:
                score += min(25, advanced_score * 8 + 5)
            # 4) 学期适配（0-25）：课程库未收录开课学期，取中性分
            if target_semester:
                score += 15
            else:
                score += 15
            if score > 0:
                scored.append((score, course))
        scored.sort(key=lambda x: (-x[0], x[1].name))
        return scored[:limit]


# ══════════════════════════════════════════════════════════════════════
# 拓扑排序规划（改编自 course_graph.py，纯算法，无 LLM）
# ══════════════════════════════════════════════════════════════════════


@dataclass
class Course:
    id: str
    name: str
    credits: float = 0
    semester: str = ""          # 秋/春/春秋/夏
    raw_prereqs: str = ""
    is_required: bool = True
    course_type: str = ""       # 必修/限选/选修
    department: str = ""


def parse_courses_from_table(markdown_text: str) -> list[Course]:
    """从培养方案原文中解析课程。

    本科培养方案原文由 PDF 转换而来（与辅修方案的 HTML 表格不同），
    课程表为多行记录：课程号独占一行，其后依次为课程名、学分、其余说明列。
    保留 HTML 表格解析作为兼容分支。
    """
    courses = _parse_html_tables(markdown_text)
    if courses:
        return courses
    return _parse_text_courses(markdown_text)


def _parse_html_tables(text: str) -> list[Course]:
    """解析 Markdown 内嵌 HTML 表格（兼容分支，本科方案数据中通常不出现）。"""
    courses = []
    rows = re.findall(r"<tr>(.*?)</tr>", text, re.DOTALL)
    if not rows:
        return courses
    current_type = ""
    for row in rows:
        cells = re.findall(r"<td[^>]*>(.*?)</td>", row, re.DOTALL)
        cells = [re.sub(r"<[^>]+>", "", c).strip() for c in cells]
        if len(cells) == 1:
            cell = cells[0]
            if "必修" in cell:
                current_type = "必修"
            elif "限选" in cell:
                current_type = "限选"
            elif "选修" in cell:
                current_type = "选修"
            continue
        if len(cells) >= 4:
            course_id = cells[0]
            if not re.match(r"^\d", course_id) and course_id not in {"新开课", "新开"}:
                continue
            name = cells[1] if len(cells) > 1 else ""
            credits = 0.0
            try:
                credits = float(cells[2]) if len(cells) > 2 else 0
            except ValueError:
                pass
            semester = cells[3] if len(cells) > 3 else ""
            raw_prereqs = cells[4] if len(cells) > 4 else ""
            if course_id or name:
                courses.append(Course(
                    id=course_id, name=name, credits=credits, semester=semester,
                    raw_prereqs=raw_prereqs,
                    is_required=(current_type == "必修"),
                    course_type=current_type,
                ))
    return courses


def _parse_text_courses(text: str) -> list[Course]:
    """解析 PDF 转换的纯文本课程记录。

    PDF 表格提取为多行记录，每列成为单独一行：
        [课程号行]    10421263
        [课程名行]    微积分C(1)
        [学分行]      3
        [其余说明行]  必修 / 秋季 / ...
    相邻两个独占一行的 8 位课程号之间为一个课程记录。
    """
    courses = []
    current_type = ""
    lines = text.split("\n")
    section_pat = re.compile(r"(必修|限选|选修|专业核心|专业基础)")
    id_pattern = re.compile(r"^\d{8}$")

    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        sm = section_pat.search(stripped)
        if sm and len(stripped) < 30:
            word = sm.group(1)
            if word in ("必修", "专业核心", "专业基础"):
                current_type = "必修"
            elif word == "限选":
                current_type = "限选"
            elif word == "选修":
                current_type = "选修"
            i += 1
            continue
        if id_pattern.match(stripped):
            course_id = stripped
            record_lines = []
            j = i + 1
            while j < len(lines):
                nxt = lines[j].strip()
                if id_pattern.match(nxt):
                    break
                if nxt:
                    record_lines.append(nxt)
                j += 1
            if record_lines:
                name = record_lines[0] if len(record_lines) > 0 else ""
                credits = 0.0
                if len(record_lines) > 1:
                    try:
                        credits = float(record_lines[1])
                    except ValueError:
                        pass
                rest_lines = record_lines[2:] if len(record_lines) > 2 else []
                semester = ""
                raw_prereqs = ""
                for rl in rest_lines:
                    if any(s in rl for s in ("春", "秋", "夏")):
                        if not semester:
                            semester = rl
                    else:
                        if raw_prereqs:
                            raw_prereqs += " " + rl
                        else:
                            raw_prereqs = rl
                if name and len(name) < 80:
                    courses.append(Course(
                        id=course_id, name=name, credits=credits,
                        semester=semester, raw_prereqs=raw_prereqs,
                        is_required=(current_type == "必修"),
                        course_type=current_type or "必修",
                    ))
            i = j
            continue
        i += 1
    return courses


def build_prerequisite_graph(courses: list[Course]):
    """构建课程先修关系 DAG，返回 (邻接表, 课程名→Course 映射)。"""
    adj: dict[str, set[str]] = {}
    course_map: dict[str, Course] = {}
    for c in courses:
        if c.name:
            course_map[c.name] = c
            adj.setdefault(c.name, set())
    for c in courses:
        if not c.raw_prereqs or not c.name:
            continue
        prereq_text = re.sub(r"<[^>]+>", "", c.raw_prereqs)
        prereq_text = re.sub(r"[、，,、]", " ", prereq_text)
        for other_name in course_map:
            if other_name != c.name and other_name in prereq_text:
                adj.setdefault(c.name, set()).add(other_name)
    return adj, course_map


def _course_offered_in_semester(course: Course, target_sem: str) -> bool:
    if not course.semester:
        return True
    sem = course.semester
    if "春秋" in sem or ("春" in sem and "秋" in sem):
        return target_sem in ("春", "秋")
    if "夏" in sem:
        return target_sem == "夏"
    if "秋" in sem:
        return target_sem == "秋"
    if "春" in sem:
        return target_sem == "春"
    return True


def topological_sort(courses: list[Course], credit_cap: float = 25.0) -> list[list[Course]]:
    """按先修关系拓扑排序生成按学期的修读计划（秋/春/夏 循环）。

    credit_cap：每学期学分上限。主修课程表规模大（90-180 门），若不限制，
    几乎所有课程会挤入第一个学期；达到上限后剩余已就绪课程顺延到下一学期。
    """
    adj, course_map = build_prerequisite_graph(courses)
    in_degree: dict[str, int] = {name: 0 for name in adj}
    for name, prereqs in adj.items():
        for prereq in prereqs:
            if prereq in in_degree:
                in_degree[name] += 1
    queue = deque(n for n, degree in in_degree.items()
                  if degree == 0 and n in course_map)

    plan: list[list[Course]] = []
    taken: set[str] = set()
    remaining: set[str] = set(course_map.keys())
    semester_cycle = ["秋", "春", "夏"]
    semester_idx = 0
    empty_streak = 0

    while remaining:
        target_sem = semester_cycle[semester_idx % len(semester_cycle)]
        if not queue:
            newly_ready = [n for n in remaining
                           if n not in taken and in_degree.get(n, 0) == 0]
            queue.extend(newly_ready)

        semester_courses: list[Course] = []
        deferred: deque[str] = deque()
        semester_credits = 0.0
        while queue:
            name = queue.popleft()
            if name in taken or name not in course_map:
                continue
            course = course_map[name]
            if course.semester and not _course_offered_in_semester(course, target_sem):
                if name not in deferred:
                    deferred.append(name)
                continue
            # 学分均衡：本学期待排学分已达上限时，把课程顺延到下一学期
            if (credit_cap and semester_credits > 0
                    and semester_credits + course.credits > credit_cap):
                if name not in deferred:
                    deferred.append(name)
                continue
            semester_courses.append(course)
            semester_credits += course.credits
            taken.add(name)
            remaining.discard(name)
            for other_name, prereqs in adj.items():
                if name in prereqs and other_name in in_degree:
                    in_degree[other_name] -= 1
                    if in_degree[other_name] == 0 and other_name not in taken:
                        queue.append(other_name)
        queue.extend(n for n in deferred if n not in taken)

        if semester_courses:
            plan.append(semester_courses)
            empty_streak = 0
        else:
            empty_streak += 1
            if empty_streak >= len(semester_cycle):
                unscheduled = [course_map[n] for n in remaining
                               if n in course_map and n not in taken]
                if unscheduled:
                    plan.append(unscheduled)
                break
        semester_idx += 1
    return plan


def format_plan(plan: list[list[Course]], student_grade: str = "大一") -> str:
    """把拓扑排序结果格式化为按学期排列的表格。"""
    grade_map = {"大一": 1, "大二": 2, "大三": 3, "大四": 4}
    current_grade_num = grade_map.get(student_grade, 1)
    semester_labels = ["秋", "春", "夏"]
    grade_names = ["大一", "大二", "大三", "大四"]

    lines = ["📋 **先修关系拓扑排序生成的参考修读计划**\n"]
    lines.append("| 学期 | 课程名称 | 学分 | 类型 |")
    lines.append("|------|----------|------|------|")
    for i, semester_courses in enumerate(plan):
        year_idx = current_grade_num - 1 + (i // 3)
        if year_idx >= 4:
            break
        lines.append(f"| **{grade_names[year_idx]} {semester_labels[i % 3]}** | | | |")
        for course in semester_courses:
            lines.append(f"| | {course.name} | {course.credits} | {course.course_type} |")
        lines.append("| | | | |")
    lines.append("\n*注：此计划由课程先修关系 DAG 拓扑排序生成，考虑了开课学期约束，"
                 "并按每学期约 25 学分做了均衡。方案原文中多数课程未标注开课学期，"
                 "先修关系部分来自课程库补全（补全覆盖不全），"
                 "请结合培养方案原文与学生实际逐课核对调整。*")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════
# 约束排课（简化自 scheduler.py，无时间冲突数据）
# ══════════════════════════════════════════════════════════════════════

SEMESTER_CYCLE = ("秋", "春", "夏")
GRADE_LABELS = ["大一", "大二", "大三", "大四", "大五"]
GRADE_TO_NUM = {"大一": 0, "大二": 1, "大三": 2, "大四": 3, "大五": 4}
SEM_TO_OFFSET = {"秋": 0, "春": 1, "夏": 2}

# 排课约束（同 scheduler.py ScheduleConstraints）
MAX_CREDITS = 25          # 每学期学分软上限
HARD_MAX_CREDITS = 30     # 硬上限
TOTAL_SEMESTERS = 8       # 规划总学期数
BAOYAN_CORE_BY = 6        # 保研核心课应在第 7 学期（大四秋）前完成


def _parse_program_requirements(raw_text: str) -> tuple[list[str], list[str]]:
    """从方案原文提取必修/选修课程号列表（按上下文关键词判定）。"""
    required_ids: list[str] = []
    elective_ids: list[str] = []
    seen: set[str] = set()
    for m in re.finditer(r"(\d{8})", raw_text):
        cid = m.group(1)
        if cid in seen:
            continue
        seen.add(cid)
        start = max(0, m.start() - 200)
        context = raw_text[start:m.end() + 50]
        if re.search(r"(必修|专业核心|专业基础|通识必修)", context):
            required_ids.append(cid)
        elif re.search(r"(选修|限选|任选|通识选修)", context):
            elective_ids.append(cid)
        else:
            required_ids.append(cid)
    return required_ids, elective_ids


def _extract_course_from_text(cid: str, raw_text: str) -> tuple[str, float, str]:
    """从方案原文中提取课程名、学分、开课学期（课程库缺失时的回退）。"""
    idx = raw_text.find(cid)
    if idx < 0:
        return (cid, 0, "")
    lines = raw_text[idx + len(cid):].strip().split("\n")
    name = cid
    credits = 0.0
    semester = ""
    if lines:
        candidate = lines[0].strip()
        if candidate and not candidate.isdigit() and len(candidate) < 80:
            name = candidate
        if len(lines) > 1:
            try:
                credits = float(lines[1].strip())
            except ValueError:
                pass
        for line in lines[2:6]:
            s = line.strip()
            if any(kw in s for kw in ("秋", "春", "夏", "春秋")):
                if not semester:
                    semester = s[:10]
                break
    return (name, credits, semester)


def _build_graph_courses(required_ids: list[str], elective_ids: list[str],
                         raw_text: str, catalog: CourseCatalog) -> list[Course]:
    """构造排课用课程对象：课程库优先，方案原文回退。"""
    courses: list[Course] = []
    for cid, is_required in [(c, True) for c in required_ids] + \
                            [(c, False) for c in elective_ids]:
        rec = catalog._by_id.get(cid)
        if rec and rec.name and rec.name != cid:
            name, credits, prereqs = rec.name, rec.credits or 0, rec.prerequisites or ""
            semester = ""
        else:
            name, credits, semester = _extract_course_from_text(cid, raw_text)
            prereqs = ""
        courses.append(Course(
            id=cid, name=name, credits=credits, semester=semester,
            raw_prereqs=prereqs, is_required=is_required,
            course_type="必修" if is_required else "选修",
        ))
    return courses


def _constrained_topological_sort(courses: list[Course], start_offset: int,
                                  total_semesters: int) -> tuple[list[list[Course]], list[Course]]:
    """带开课学期约束与学分节奏的拓扑排序。

    必修课学期上限 30 学分、选修课软上限 25 学分；达到上限的课程顺延到下一学期。
    返回 (按学期排布结果, 未能排入的课程)。
    """
    adj, course_map = build_prerequisite_graph(courses)
    in_degree: dict[str, int] = {}
    for name in adj:
        in_degree[name] = 0
    for name, prereqs in adj.items():
        for prereq in prereqs:
            if prereq in in_degree:
                in_degree[name] += 1
    queue = deque(n for n, deg in in_degree.items()
                  if deg == 0 and n in course_map)

    plan: list[list[Course]] = []
    taken: set[str] = set()
    all_names: set[str] = set(course_map.keys())

    for sem_idx in range(total_semesters):
        target_sem = SEMESTER_CYCLE[(start_offset + sem_idx) % 3]
        if not queue:
            newly_ready = [n for n in all_names
                           if n not in taken and in_degree.get(n, 0) == 0]
            queue.extend(newly_ready)
        semester_courses: list[Course] = []
        deferred: deque[str] = deque()
        semester_credits = 0.0
        while queue:
            name = queue.popleft()
            if name in taken or name not in course_map:
                continue
            course = course_map[name]
            if course.semester and not _course_offered_in_semester(course, target_sem):
                deferred.append(name)
                continue
            # 学分节奏：必修课学期上限30、选修课软上限25，超限顺延到下一学期
            if semester_credits > 0:
                cap = HARD_MAX_CREDITS if course.is_required else MAX_CREDITS
                if semester_credits + course.credits > cap:
                    deferred.append(name)
                    continue
            semester_courses.append(course)
            semester_credits += course.credits
            taken.add(name)
            for other_name, prereqs in adj.items():
                if name in prereqs and other_name in in_degree:
                    in_degree[other_name] -= 1
                    if in_degree[other_name] == 0 and other_name not in taken:
                        queue.append(other_name)
        while deferred:
            n = deferred.popleft()
            if n not in taken:
                queue.append(n)
        if semester_courses:
            plan.append((start_offset + sem_idx, semester_courses))
        if len(taken) >= len(all_names):
            break
    unscheduled = [course_map[n] for n in all_names if n not in taken]
    return plan, unscheduled


def _semester_label(abs_idx: int) -> str:
    """绝对学期索引（0=大一秋）→ 学期标签。"""
    year = abs_idx // 3
    if 0 <= year < len(GRADE_LABELS):
        return f"{GRADE_LABELS[year]}{SEMESTER_CYCLE[abs_idx % 3]}"
    return f"第{year + 1}年{SEMESTER_CYCLE[abs_idx % 3]}"


def _apply_baoyan_warnings(scheduled: list[tuple[int, list[Course]]], gpa: float,
                           goals: list[str]) -> list[str]:
    """保研约束提醒（目标含保研/读研时）。"""
    warnings: list[str] = []
    if not any(g in goals for g in ("保研", "读研")):
        return warnings
    late_required: list[str] = []
    for abs_idx, sem_courses in scheduled:
        if abs_idx >= BAOYAN_CORE_BY:
            late_required.extend(c.name for c in sem_courses if c.is_required)
    if late_required:
        warnings.append(
            f"[保研提醒] 以下必修课安排在第{BAOYAN_CORE_BY + 1}学期（大四秋）或之后"
            f"（保研申请通常在大四秋），建议提前修读：{'、'.join(late_required[:5])}")
    if 0 < gpa < 3.0:
        warnings.append("[保研提醒] 当前 GPA 偏低，保研通常要求 3.0 以上，请重点关注本学期课程成绩。")
    elif 0 < gpa < 3.5:
        warnings.append("[保研建议] 保研竞争激烈，建议将 GPA 提升至 3.5 以上。")
    return warnings


def _format_schedule(scheduled: list[tuple[int, list[Course]]],
                     major: str, grade: str, program: TrainingProgram,
                     goals: list[str], warnings: list[str]) -> str:
    """把排课结果格式化为 Markdown 表格。"""
    total_required = sum(c.credits for _, sem in scheduled for c in sem if c.is_required)
    total_elective = sum(c.credits for _, sem in scheduled for c in sem if not c.is_required)
    course_count = sum(len(sem) for _, sem in scheduled)

    lines = [
        "═══════════════════════════════════",
        f"  培养方案：{program.name}",
        f"  学生专业：{major}　年级：{grade}",
        f"  总学分规划：必修{total_required:.0f} + 选修{total_elective:.0f} "
        f"= {total_required + total_elective:.0f}学分",
        f"  课程总数：{course_count}门　规划学期：{len(scheduled)}个",
    ]
    if goals:
        lines.append(f"  目标：{'、'.join(goals)}")
    if warnings:
        lines.append("  ────────────────────────────────")
        lines.extend(f"  {w}" for w in warnings)
    lines.append("═══════════════════════════════════")

    for abs_idx, sem_courses in scheduled:
        label = _semester_label(abs_idx)
        total = sum(c.credits for c in sem_courses)
        req = sum(1 for c in sem_courses if c.is_required)
        ele = sum(1 for c in sem_courses if not c.is_required)
        lines.append(f"\n## {label}")
        lines.append(f"> 必修{req}门 + 选修{ele}门　共{total:.0f}学分")
        lines.append("")
        lines.append("| 课程号 | 课程名称 | 学分 | 类型 |")
        lines.append("|--------|----------|------|------|")
        for c in sem_courses:
            lines.append(
                f"| {c.id} | {c.name} | {c.credits} | "
                f"{'[必修]' if c.is_required else '[选修]'} |")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════
# 查询工具（改编自 tools.py，去掉语义搜索与 Multi-Agent）
# ══════════════════════════════════════════════════════════════════════


def tool_list_programs(flat: bool = False) -> str:
    programs = load_programs()
    if flat:
        return "清华大学2025级本科培养方案列表：\n" + "\n".join(
            f"  {i + 1}. {m.name}（{m.department}）" for i, m in enumerate(programs))
    grouped: dict[str, list[str]] = defaultdict(list)
    for m in programs:
        grouped[m.department or "其他院系"].append(m.name)
    lines = ["清华大学2025级本科培养方案列表（按院系分组）："]
    for dept, names in sorted(grouped.items()):
        lines.append(f"\n【{dept}】")
        lines.extend(f"  - {n}" for n in names)
    lines.append(f"\n共 {len(programs)} 个培养方案。")
    return "\n".join(lines)


def tool_search_programs(keyword: str) -> str:
    if not keyword.strip():
        return "请提供专业名称、院系或方向关键词后再搜索。"
    results = search_programs(keyword, load_programs())
    if not results:
        return f"未找到与 '{keyword}' 相关的培养方案。"
    shown = results[:10]
    lines = [f"找到 {len(results)} 个相关培养方案："]
    if len(results) > len(shown):
        lines[0] += (f"\n（仅显示前 {len(shown)} 个，可换更精确的关键词或院系名缩小范围）")
    for m in shown:
        lines.append(f"\n【{m.name}】({m.department})")
        lines.append(f"  学分要求：{m.total_credits or '见培养方案原文'}")
        if m.degree:
            lines.append(f"  授予学位：{m.degree[:100]}")
        if m.duration:
            lines.append(f"  学制：{m.duration[:100]}")
        if m.contact:
            lines.append(f"  咨询电话：{m.contact}")
    return "\n".join(lines)


def tool_get_program_detail(name: str) -> str:
    programs = load_programs()
    prog = get_program_by_name(name, programs)
    if not prog:
        return f"未找到培养方案: {name}"
    parts = [
        f"【{prog.name}】",
        f"开设院系：{prog.department}",
        f"学分要求：{prog.total_credits or '见培养方案原文'}",
    ]
    if prog.degree:
        parts.append(f"授予学位：{prog.degree[:100]}")
    if prog.duration:
        parts.append(f"学制：{prog.duration[:100]}")
    if prog.prerequisites:
        parts.append(f"先修课程：{prog.prerequisites}")
    if prog.major_restrictions:
        parts.append(f"招生说明：{prog.major_restrictions}")
    if prog.contact:
        parts.append(f"咨询电话：{prog.contact}")
    dup = _same_name_duplicates(prog, programs)
    if dup:
        parts.append(f"⚠️ 同名培养方案：{'、'.join(m.department for m in dup)}，"
                     f"如非所需院系请用 search_programs 确认")
    parts.append(f"\n--- 培养方案详情 ---\n{prog.raw_text[:3000]}")
    return "\n".join(parts)


def tool_check_requirements(major: str, program_name: str) -> str:
    prog = get_program_by_name(program_name, load_programs())
    if not prog:
        return f"未找到培养方案: {program_name}"
    return (
        f"【培养方案要求】专业：{major} → 培养方案：{prog.name}\n"
        f"学分要求：{prog.total_credits or '见培养方案原文'}\n"
        f"学位：{prog.degree[:100] or '见培养方案原文'}\n"
        f"学制：{prog.duration[:100] or '见培养方案原文'}\n"
        f"说明：{prog.major_restrictions or '见培养方案原文详情'}\n"
        f"注：以上字段为自动提取，可能与方案原文有出入，重要数字请以 get_program_detail 的方案原文为准。"
    )


def tool_search_courses(keyword: str) -> str:
    if not keyword.strip():
        return "请提供课程号、课程名称或主题关键词后再搜索。"
    catalog = CourseCatalog(load_courses())
    all_matches = catalog.search(keyword, limit=1000)
    if not all_matches:
        return f'未在已收录的课程资料中找到与"{keyword}"相关的课程。'
    courses = all_matches[:5]
    lines = [f"找到 {len(courses)} 门相关课程："]
    if len(all_matches) > len(courses):
        lines[0] = (f"找到 {len(all_matches)} 门相关课程（仅显示前 {len(courses)} 门，"
                    f"可换更精确的关键词或课程号）：")
    for course in courses:
        programs = "、".join(
            str(p.get("program", "")).replace("专业辅修培养方案", "")
            for p in course.minor_programs[:3])
        lines.append(
            f"\n【{course.name}】课程号：{course.id}｜{course.department}｜"
            f"{course.credits if course.credits is not None else '未知'} 学分")
        if programs:
            lines.append(f"  关联培养方案：{programs}")
        if course.prerequisites:
            lines.append(f"  先修要求：{course.prerequisites[:200]}")
        if course.description:
            lines.append(f"  内容摘要：{course.description[:240]}")
    return "\n".join(lines)


def tool_get_course_detail(identifier: str) -> str:
    catalog = CourseCatalog(load_courses())
    course = catalog.find(identifier)
    if not course:
        matches = catalog.search(identifier, limit=2) if identifier.strip() else []
        if matches:
            return "课程名称不够明确，请从以下候选中指定一门：" + "、".join(
                f"{item.name}（{item.id}）" for item in matches)
        return f"未找到课程：{identifier}"
    programs = "、".join(
        str(p.get("program", "")).replace("专业辅修培养方案", "")
        for p in course.minor_programs) or "未标注"
    parts = [
        f"【{course.name}】",
        f"课程号：{course.id}",
        f"开课单位：{course.department or '未标注'}",
        f"学分：{course.credits if course.credits is not None else '未标注'}",
        f"总学时：{course.total_hours if course.total_hours is not None else '未标注'}",
        f"关联培养方案：{programs}",
    ]
    missing = []
    for label, value in (
        ("先修要求", course.prerequisites),
        ("课程内容", course.description),
        ("教学目标", course.objectives),
        ("预期学习成效", course.expected_outcomes),
        ("考核方式", course.assessment_method),
        ("成绩构成", course.grade_breakdown),
        ("教材及参考书", course.textbooks),
        ("课程负责人", course.instructor),
    ):
        if value:
            parts.append(f"\n{label}：{value[:1200]}")
        else:
            missing.append(label)
    if missing:
        parts.append(f"\n注：资料库未收录：{'、'.join(missing)}。")
    return "\n".join(parts)


def tool_list_program_courses(program_name: str) -> str:
    programs = load_programs()
    prog = get_program_by_name(program_name, programs)
    if not prog:
        return f"未找到培养方案: {program_name}"
    all_courses = CourseCatalog(load_courses()).for_program(
        prog.name, department=prog.department, limit=1000)
    if not all_courses:
        return f"{prog.name} 暂无可用的详细课程资料。"
    courses = all_courses[:40]
    lines = [f"{prog.name} 课程库中已收录详细资料的课程（共 {len(all_courses)} 门）："]
    if len(all_courses) > len(courses):
        lines[0] += f"\n（仅显示前 {len(courses)} 门，完整课程表见培养方案原文）"
    lines.append("注：此列表来自课程库的关联字段，可能与培养方案原文课程表存在偏差，"
                 "课程号与学分请以方案原文为准。")
    for c in courses:
        lines.append(
            f"- {c.name}（{c.id}，{c.credits if c.credits is not None else '未知'} "
            f"学分，{c.department}）")
    return "\n".join(lines)


def tool_recommend_courses(major: str = "", grade: str = "",
                           interests: str = "", semester: str = "") -> str:
    if not major and not interests:
        return "请提供你的专业或兴趣方向，以便推荐适合的课程。"
    results = CourseCatalog(load_courses()).recommend(
        major=major, grade=grade, interests=interests,
        target_semester=semester, limit=10)
    if not results:
        return f"未找到与你的条件匹配的课程。请尝试调整关键词或专业信息。"
    lines = [f"基于你的专业（{major or '未指定'}）、年级（{grade or '未指定'}）"]
    if interests:
        lines[0] += f"和兴趣（{interests}）"
    lines[0] += f"，推荐以下 {len(results)} 门课程："
    for score, course in results:
        type_info = f"｜{course.course_type}" if course.course_type else ""
        lines.append(
            f"\n【{course.name}】课程号：{course.id}｜"
            f"{course.department}｜{course.credits or '?'}学分{type_info}｜推荐度：{score}")
        if course.prerequisites:
            lines.append(f"  先修要求：{course.prerequisites[:150]}")
        if course.description:
            lines.append(f"  内容摘要：{course.description[:200]}")
    lines.append("\n注：推荐结果基于课程库资料打分，开课学期数据未收录，"
                 "实际选课请以选课系统为准。")
    return "\n".join(lines)


def _supplement_prereqs_from_catalog(courses: list[Course]) -> list[Course]:
    """用课程库的先修信息补充培养方案表中缺失的先修关系。"""
    catalog = CourseCatalog(load_courses())
    by_key = {}
    for c in catalog.courses:
        by_key.setdefault(_normalize(c.name), c)
        if c.id:
            by_key.setdefault(_normalize(c.id), c)
    placeholder = ("", "适用", "无", "无适用", "-", "—")
    for course in courses:
        if course.raw_prereqs and course.raw_prereqs not in placeholder:
            continue
        lookup = by_key.get(_normalize(course.name)) or by_key.get(_normalize(course.id))
        if not lookup or not lookup.prerequisites:
            continue
        prereq = re.sub(r"适用$", "", lookup.prerequisites).strip()
        prereq = re.sub(r"^(无|无适用)[、,，]?", "", prereq).strip()
        if prereq and prereq not in placeholder:
            course.raw_prereqs = prereq
    return courses


def _find_thesis_requirements(raw_text: str) -> list[str]:
    """从方案原文中识别毕业论文/综合论文训练等要求。

    部分方案的论文训练没有课程号（不在课程表中），解析不到课程记录，
    此处按原文行扫描补齐（如"综合论文训练 9学分 必修"）。
    """
    items: list[str] = []
    for m in re.finditer(r"(综合论文训练|毕业论文|毕业设计|毕业创作|毕业综合训练)", raw_text):
        tail = raw_text[m.end():m.end() + 60].split("\n")[0].strip()
        text = m.group(1)
        if "学分" in tail:
            text += f"（{tail[:15]}）"
        if text not in items:
            items.append(text)
    return items[:6]


def tool_plan(major: str, grade: str, program_name: str) -> str:
    programs = load_programs()
    prog = get_program_by_name(program_name, programs)
    if not prog:
        return f"未找到培养方案: {program_name}"
    courses = parse_courses_from_table(prog.raw_text)
    if not courses:
        return f"{prog.name} 培养方案中未能解析出课程表，请结合方案原文（get_program_detail）制定计划。"
    # 综合论文训练等论文类课程不参与学期排布，单独标注在毕业学期
    thesis = [c for c in courses if "论文" in c.name]
    regular = [c for c in courses if "论文" not in c.name]
    _supplement_prereqs_from_catalog(regular)
    algo_text = format_plan(topological_sort(regular), grade)
    thesis_notes: list[str] = []
    if thesis:
        thesis_notes.append("、".join(
            f"{t.name}{t.credits}学分" if t.credits else t.name for t in thesis))
    # 原文中无课程号的论文训练要求（解析不到课程记录时）
    thesis_notes.extend(
        t for t in _find_thesis_requirements(prog.raw_text)
        if not any(t.split("（")[0] in n for n in thesis_notes))
    if thesis_notes:
        algo_text += f"\n\n**毕业学期**：完成{'、'.join(thesis_notes[:4])}。"
    parts = [
        f"## 📋 {prog.name} — 修读计划（{major} · {grade}）\n",
        algo_text,
    ]
    if prog.contact:
        parts.append(f"\n📞 咨询电话：{prog.contact}")
    if prog.degree:
        parts.append(f"🎓 授予学位：{prog.degree[:100]}")
    if prog.duration:
        parts.append(f"⏱ 学制：{prog.duration[:100]}")
    parts.append(
        "\n*注：此为算法生成的参考计划（先修关系已用课程库资料补全，但补全覆盖不全），"
        "请结合学生实际情况、开课学期变化与培养方案最新要求逐课核对调整。*")
    return "\n".join(parts)


def tool_schedule(major: str, grade: str, program_name: str,
                  completed_courses: str = "", gpa: str = "",
                  goals: str = "", target_semester: str = "秋") -> str:
    """约束优化排课：先修关系 + 开课学期 + 学分上限 + 已修排除 + 保研提醒。

    上课时间冲突检测依赖课程时间数据，该数据尚未收录，排课结果不包含具体上课时段。
    """
    programs = load_programs()
    prog = get_program_by_name(program_name, programs)
    if not prog:
        return f"未找到培养方案: {program_name}"

    try:
        gpa_val = float(gpa) if gpa and gpa.strip() else 0.0
    except ValueError:
        gpa_val = 0.0
    goal_list = [g.strip() for g in re.split(r"[、,，]", goals) if g.strip()]
    completed = [c.strip() for c in re.split(r"[、,，]", completed_courses) if c.strip()]

    # 1) 从方案原文解析必修/选修课程号
    required_ids, elective_ids = _parse_program_requirements(prog.raw_text)
    catalog = CourseCatalog(load_courses())
    if not required_ids and not elective_ids:
        matched = catalog.for_program(prog.name, department=prog.department, limit=80)
        required_ids = [c.id for c in matched if any(
            "必修" in str(mp) or "核心" in str(mp) for mp in c.minor_programs)]
        elective_ids = [c.id for c in matched if c.id not in required_ids]

    # 2) 构造课程对象（课程库优先，方案原文回退解析课程名/学分）
    graph_courses = _build_graph_courses(
        required_ids, elective_ids, prog.raw_text, catalog)
    # 3) 排除已修课程（接受课程号或课程名，与解析出的课程名做规范化比对）
    completed_norm = {_normalize(c) for c in completed}
    graph_courses = [c for c in graph_courses
                     if _normalize(c.id) not in completed_norm
                     and _normalize(c.name) not in completed_norm]
    # 论文类课程不参与学期排布，单独标注在毕业学期
    thesis = [c for c in graph_courses
              if "论文" in c.name or "毕业设计" in c.name or "毕业创作" in c.name]
    graph_courses = [c for c in graph_courses if c not in thesis]
    if not graph_courses:
        return ("未能解析到可排课程，请检查培养方案数据（可先用 get_program_detail 查看原文）。")
    grade_num = GRADE_TO_NUM.get(grade, 1)
    start_sem = SEM_TO_OFFSET.get(target_semester, 0)
    start_offset = start_sem + grade_num * 3

    scheduled, unscheduled = _constrained_topological_sort(
        graph_courses, start_offset, TOTAL_SEMESTERS)

    # 4) 保研约束与未排入课程提示
    warnings: list[str] = []
    warnings.extend(_apply_baoyan_warnings(scheduled, gpa_val, goal_list))
    if unscheduled:
        warnings.append(
            "以下课程未能排入规划学期（学分上限或先修约束），建议延后安排或与院系确认："
            + "、".join(f"{c.name}（{c.id}）" for c in unscheduled[:8])
            + (f"等共{len(unscheduled)}门" if len(unscheduled) > 8 else ""))
    if thesis:
        warnings.append("论文类课程不参与学期排布，请在毕业学期完成："
                        + "、".join(f"{c.name}" for c in thesis))

    if not scheduled:
        return "未能生成排课结果：已修课程可能覆盖了全部课程，或方案解析失败。"

    lines = [_format_schedule(scheduled, major, grade, prog, goal_list, warnings)]
    if any(g in goal_list for g in ("保研", "读研")):
        lines.append("\n### [保研] 专项提示")
        lines.append("- 大四秋季学期（9月）是保研申请关键期，此前需完成所有必修课并取得较好成绩")
        lines.append("- 英语六级（CET-6）建议 425 分以上，部分院系有更高要求")
        lines.append("- 科研/竞赛经历是保研加分项，建议大二大三积极参与")
        lines.append("- 联系导师通常在大三下-大四秋，提前准备个人陈述和简历")
    lines.append("\n*注：排课考虑了先修关系、开课学期（方案原文多未标注，按无约束处理）、"
                 "每学期学分上限（必修软上限30/选修软上限25）。方案表格包含各方向/轨道的"
                 "全部课程，学分合计会超过方案总学分，请按学生实际方向与限选要求筛选。"
                 "上课时间冲突检测依赖的课程时间数据尚未收录，具体上课时段请以选课系统为准。*")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════
# CLI 入口
# ══════════════════════════════════════════════════════════════════════

def main(argv: list[str]) -> int:
    import argparse
    parser = argparse.ArgumentParser(
        prog="advisor.py",
        description="清华大学本科培养方案助手 — 数据查询 CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("list_programs", help="列出所有本科培养方案（按院系分组）")
    p.add_argument("--flat", action="store_true", help="不分组的扁平列表")

    sub.add_parser("search_programs", help="按关键词搜索培养方案").add_argument(
        "keyword", help="专业名称/院系/方向关键词")
    sub.add_parser("get_program_detail", help="获取培养方案详细内容").add_argument(
        "name", help="培养方案名称")
    p = sub.add_parser("check_requirements", help="检查某专业的基本要求")
    p.add_argument("major", help="学生的专业")
    p.add_argument("program_name", help="培养方案名称")
    sub.add_parser("search_courses", help="按课程号/名称/主题搜索课程").add_argument(
        "keyword", help="例如：机器学习、数据结构、30000833")
    sub.add_parser("get_course_detail", help="获取课程详细资料").add_argument(
        "identifier", help="精确课程号或明确课程名称")
    sub.add_parser("list_program_courses", help="列出某培养方案已收录详细资料的课程").add_argument(
        "program_name", help="培养方案名称")

    p = sub.add_parser("recommend_courses", help="按专业/年级/兴趣推荐课程")
    p.add_argument("major", help="学生的专业")
    p.add_argument("grade", help="年级（大一/大二/大三/大四）")
    p.add_argument("interests", help="兴趣方向（如：人工智能、经济学、建筑设计）")
    p.add_argument("--semester", default="", help="目标学期（秋/春/夏），可选")

    p = sub.add_parser("plan", help="生成按学期的修读计划（先修关系拓扑排序）")
    p.add_argument("major", help="学生的专业")
    p.add_argument("grade", help="当前年级（大一/大二/大三/大四）")
    p.add_argument("program_name", help="培养方案名称")

    p = sub.add_parser("schedule", help="约束优化排课（先修/学期/学分上限/已修排除）")
    p.add_argument("major", help="学生的专业")
    p.add_argument("grade", help="当前年级（大一/大二/大三/大四）")
    p.add_argument("program_name", help="培养方案名称")
    p.add_argument("--completed", default="", help="已修课程，逗号分隔（如：微积分,线性代数,10421263）")
    p.add_argument("--gpa", default="", help="当前GPA（可选，用于保研评估）")
    p.add_argument("--goals", default="", help="目标，逗号分隔（如：保研,出国）")
    p.add_argument("--start", default="秋", help="开始排课学期（秋/春/夏），默认秋")

    args = parser.parse_args(argv)
    cmd = args.command
    try:
        if cmd == "list_programs":
            print(tool_list_programs(flat=args.flat))
        elif cmd == "search_programs":
            print(tool_search_programs(args.keyword))
        elif cmd == "get_program_detail":
            print(tool_get_program_detail(args.name))
        elif cmd == "check_requirements":
            print(tool_check_requirements(args.major, args.program_name))
        elif cmd == "search_courses":
            print(tool_search_courses(args.keyword))
        elif cmd == "get_course_detail":
            print(tool_get_course_detail(args.identifier))
        elif cmd == "list_program_courses":
            print(tool_list_program_courses(args.program_name))
        elif cmd == "recommend_courses":
            print(tool_recommend_courses(args.major, args.grade, args.interests,
                                         args.semester))
        elif cmd == "plan":
            print(tool_plan(args.major, args.grade, args.program_name))
        elif cmd == "schedule":
            print(tool_schedule(args.major, args.grade, args.program_name,
                                completed_courses=args.completed, gpa=args.gpa,
                                goals=args.goals, target_semester=args.start))
    except FileNotFoundError as exc:
        print(f"错误：缺少数据文件 {exc.filename}，请确认 skill 的 data/ 目录完整。",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
