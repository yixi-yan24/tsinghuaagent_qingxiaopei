"""
智能排课引擎 — 基于约束优化的课程表自动生成

支持约束：
- 先修关系 DAG
- 开课学期（秋/春/夏）
- 学分上限（默认25/学期，硬上限30）
- 保研要求（核心课前置、GPA相关课程优先）
- 已修课程排除
- 培养方案必修课全覆盖
"""

import re
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict, deque

import os, json

from .course_catalog import CourseCatalog, CourseRecord
from .course_graph import Course as GraphCourse, build_prerequisite_graph, topological_sort, _course_offered_in_semester
from .data_loader import TrainingProgram

SCHEDULE_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "course_schedule.json")


# ═══════════════════════════════════════════════════════════════════════
# Data models
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class TimeSlot:
    """课程时间段"""
    day: int          # 1=周一 ... 7=周日
    start_period: int  # 开始节次 1-6
    end_period: int    # 结束节次 1-6
    weeks: str = ""    # 周次描述 "1-16" / "1,3,5,7,9,11,13,15"

    def overlaps(self, other: "TimeSlot") -> bool:
        """Check if two time slots overlap (same day + overlapping periods)."""
        if self.day != other.day:
            return False
        return (self.start_period <= other.end_period and
                other.start_period <= self.end_period)

    @staticmethod
    def from_dict(data: dict) -> "TimeSlot":
        return TimeSlot(
            day=data.get("day", 1),
            start_period=data.get("start_period", 1),
            end_period=data.get("end_period", 1),
            weeks=str(data.get("weeks", "")),
        )


@dataclass
class CourseOption:
    """课程的一个班次（含时间段组合）"""
    class_id: str = ""          # "A班" / "B班" / "1班"
    instructor: str = ""        # 授课教师
    slots: list = field(default_factory=list)  # TimeSlot列表

    def conflicts_with(self, other_option: "CourseOption") -> bool:
        """Check if any slot in this option overlaps with any slot in the other."""
        for s1 in self.slots:
            for s2 in other_option.slots:
                if s1.overlaps(s2):
                    return True
        return False

    @staticmethod
    def from_dict(data: dict) -> "CourseOption":
        return CourseOption(
            class_id=str(data.get("class_id", "")),
            instructor=str(data.get("instructor", "")),
            slots=[TimeSlot.from_dict(s) for s in (data.get("slots") or [])],
        )

@dataclass
class StudentProfile:
    """学生档案"""
    major: str = ""
    grade: str = ""             # 大一/大二/大三/大四
    completed_courses: list[str] = field(default_factory=list)  # 已修课程号 或 课程名
    gpa: float = 0.0
    goals: list[str] = field(default_factory=list)  # ["保研", "出国", "就业"...]
    target_semester_start: str = ""  # 开始排课的学期 "秋"/"春"/"夏"


@dataclass
class ScheduleConstraints:
    """排课约束"""
    max_credits_per_semester: int = 25      # 每学期学分上限
    hard_max_credits: int = 30              # 硬上限
    min_credits_per_semester: int = 15      # 下限
    total_semesters: int = 8                # 规划总学期数
    # 保研约束
    baoyan_require_core_by_semester: int = 6  # 第几学期前修完核心课(大四秋=7)
    baoyan_min_gpa_courses: int = 5         # 需要重点关注的GPA课程数
    # 清华大学学期设置
    semesters: tuple = ("秋", "春", "夏")
    # 时间冲突
    check_time_conflicts: bool = True      # 是否检查时间冲突
    max_conflicts_before_warn: int = 0     # 允许的冲突数（0=禁止任何冲突）


@dataclass
class ScheduledCourse:
    """排入课表的课程"""
    course_id: str
    course_name: str
    credits: float
    semester_index: int       # 0-based semester index
    semester_label: str       # "大一秋" / "大二春" ...
    is_required: bool = True
    reason: str = ""          # 排课理由
    time_slots: list = field(default_factory=list)  # TimeSlot列表（已废弃，用chosen_option）
    chosen_option: object = None  # CourseOption | None — 排课时选中的班次


@dataclass
class ScheduleResult:
    """排课结果"""
    student: StudentProfile
    program_name: str
    semesters: list[list[ScheduledCourse]]  # 每学期一个列表
    summary: str = ""                       # 总结文本
    warnings: list[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════
# Semester helper
# ═══════════════════════════════════════════════════════════════════════

SEMESTER_CYCLE = ("秋", "春", "夏")
GRADE_LABELS = ["大一", "大二", "大三", "大四", "大五"]
GRADE_TO_NUM = {"大一": 0, "大二": 1, "大三": 2, "大四": 3, "大五": 4}


def _semester_label(grade_num: int, sem: str) -> str:
    """Return Chinese label like '大一秋', '大二春'."""
    if 0 <= grade_num < len(GRADE_LABELS):
        return f"{GRADE_LABELS[grade_num]}{sem}"
    return f"第{grade_num+1}年{sem}"


# ═══════════════════════════════════════════════════════════════════════
# Core scheduler
# ═══════════════════════════════════════════════════════════════════════

class CourseScheduler:
    """约束优化排课器"""

    def __init__(self, catalog: CourseCatalog = None,
                 constraints: ScheduleConstraints = None):
        self.catalog = catalog or CourseCatalog.load()
        self.constraints = constraints or ScheduleConstraints()
        # Cache
        self._course_index: dict[str, CourseRecord] = {}
        self._schedule_data: dict[str, list] = self._load_schedule_data()

    # ── public API ───────────────────────────────────────────────────

    def generate(self, student: StudentProfile, program: TrainingProgram) -> ScheduleResult:
        """主入口：为学生和培养方案生成推荐课表。

        Returns a ScheduleResult with semester-by-semester course assignments.
        """
        warnings: list[str] = []
        self._build_index()

        # 1) Parse required courses from program raw_text
        required_ids, elective_ids = self._parse_program_requirements(program)
        if not required_ids and not elective_ids:
            # Fallback: try to find courses by department/program match
            required_ids, elective_ids = self._infer_requirements(program)

        # 2) Remove completed courses
        completed_set = self._normalize_completed(student.completed_courses)
        remaining_required = [cid for cid in required_ids if cid not in completed_set]
        remaining_elective = [cid for cid in elective_ids if cid not in completed_set]

        if not remaining_required:
            warnings.append("所有必修课已完成，仅剩选修课待安排。")

        # 3) Build course objects for the graph
        graph_courses = self._build_graph_courses(
            remaining_required, remaining_elective, program
        )
        if not graph_courses:
            return ScheduleResult(
                student=student, program_name=program.name,
                semesters=[], warnings=["未能解析到可排课程，请检查培养方案数据。"]
            )

        # 4) Run constrained topological sort
        grade_num = GRADE_TO_NUM.get(student.grade, 1)
        start_semester = self._semester_to_offset(student.target_semester_start)
        total_semesters = self.constraints.total_semesters

        raw_semesters = self._constrained_topological_sort(
            graph_courses,
            start_semester_offset=start_semester + grade_num * 3,
            total_semesters=total_semesters,
        )

        # 5) Apply credit limits and rebalance
        scheduled_semesters, balance_warnings = self._apply_credit_limits(
            raw_semesters, grade_num, start_semester
        )
        warnings.extend(balance_warnings)

        # 6) Apply 保研 constraints
        if "保研" in student.goals or "读研" in student.goals:
            baoyan_warnings = self._apply_baoyan_constraints(
                scheduled_semesters, student, grade_num
            )
            warnings.extend(baoyan_warnings)

        # 7) Build summary
        summary = self._build_summary(
            scheduled_semesters, student, program, warnings
        )

        return ScheduleResult(
            student=student,
            program_name=program.name,
            semesters=scheduled_semesters,
            summary=summary,
            warnings=warnings,
        )

    # ── time-slot helpers ────────────────────────────────────────────

    @staticmethod
    def _load_schedule_data() -> dict[str, list]:
        """Load course time-slot data from JSON.

        Returns {course_id: [CourseOption, ...]} mapping.
        Each course can have multiple options (A班/B班/...).
        Empty dict if file doesn't exist or has no real data yet.
        """
        if not os.path.exists(SCHEDULE_DATA_PATH):
            return {}
        try:
            with open(SCHEDULE_DATA_PATH, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}

        result: dict[str, list] = {}
        for entry in data:
            cid = str(entry.get("course_id", "")).strip()
            if not cid:
                continue

            # New format: "options" list
            options_raw = entry.get("options") or []
            if options_raw:
                # Skip if this is the placeholder example (only entry with options)
                if len(data) <= 1:
                    continue
                options = [CourseOption.from_dict(o) for o in options_raw]
                if options:
                    result[cid] = options
                continue

            # Legacy format: flat "slots" list → single option
            slots_raw = entry.get("slots") or []
            if slots_raw:
                slots = [TimeSlot.from_dict(s) for s in slots_raw]
                if slots:
                    result[cid] = [CourseOption(class_id="默认", slots=slots)]

        return result

    def _get_course_options(self, course_id: str) -> list:
        """Get all time-slot options for a course. Returns [CourseOption, ...] or []."""
        return self._schedule_data.get(course_id, [])

    def _find_non_conflicting_option(self, new_course_id: str,
                                     assigned_courses: list
                                     ) -> tuple[Optional[CourseOption], list[str]]:
        """Try each option of a course, return the first one without time conflicts.

        Returns:
            (chosen_option, conflict_descriptions)
            chosen_option is None if no option is available or no time data exists.
            conflict_descriptions lists all conflicts found (for warnings).
        """
        options = self._get_course_options(new_course_id)
        if not options:
            return (None, [])  # No time data → no constraints

        # Collect all existing course options (already chose one per assigned course)
        existing_options: list[CourseOption] = []
        for assigned in assigned_courses:
            aid = getattr(assigned, 'course_id', '')
            # We need the CHOSEN option for each assigned course
            chosen = getattr(assigned, 'chosen_option', None)
            if chosen and isinstance(chosen, CourseOption):
                existing_options.append(chosen)

        if not existing_options:
            # No assigned courses have time data → first option is fine
            return (options[0], [])

        all_conflicts: list[str] = []
        for option in options:
            option_conflicts: list[str] = []
            for existing_opt in existing_options:
                if option.conflicts_with(existing_opt):
                    existing_cid = ""
                    for a in assigned_courses:
                        if getattr(a, 'chosen_option', None) is existing_opt:
                            existing_cid = getattr(a, 'course_id', '')
                            break
                    option_conflicts.append(
                        f"[时间冲突] {new_course_id}({option.class_id}) "
                        f"与 {existing_cid} 冲突 → 跳过此班次"
                    )
            if not option_conflicts:
                # This option works!
                return (option, all_conflicts + option_conflicts)
            all_conflicts.extend(option_conflicts)

        # All options conflict
        return (None, all_conflicts)

    def _check_time_conflicts(self, new_course_id: str,
                              assigned_courses: list) -> list[str]:
        """Check time conflicts — legacy wrapper, use _find_non_conflicting_option instead."""
        if not self.constraints.check_time_conflicts:
            return []
        _, conflicts = self._find_non_conflicting_option(new_course_id, assigned_courses)
        return conflicts

    # ── helpers ──────────────────────────────────────────────────────

    def _build_index(self):
        """Build course ID → CourseRecord lookup."""
        for c in self.catalog.courses:
            self._course_index[c.id] = c

    @staticmethod
    def _normalize_completed(completed: list[str]) -> set[str]:
        """Normalize completed course identifiers to a set of IDs."""
        result: set[str] = set()
        for item in completed:
            item = item.strip()
            if not item:
                continue
            # Could be "10421263" or "微积分C(1)"
            result.add(item)
        return result

    def _parse_program_requirements(self, program: TrainingProgram
                                    ) -> tuple[list[str], list[str]]:
        """Extract required/elective course IDs from program raw_text.

        Returns (required_ids, elective_ids).
        """
        text = program.raw_text
        required_ids: list[str] = []
        elective_ids: list[str] = []
        seen: set[str] = set()

        # Find 8-digit course IDs with surrounding context
        for m in re.finditer(r"(\d{8})", text):
            cid = m.group(1)
            if cid in seen:
                continue
            seen.add(cid)
            # Look at surrounding context to determine required vs elective
            start = max(0, m.start() - 200)
            end = min(len(text), m.end() + 50)
            context = text[start:end]

            if re.search(r"(必修|专业核心|专业基础|通识必修)", context):
                required_ids.append(cid)
            elif re.search(r"(选修|限选|任选|通识选修)", context):
                elective_ids.append(cid)
            else:
                # Default: add to required if it appears in a table with "必修" header
                required_ids.append(cid)

        return required_ids, elective_ids

    def _infer_requirements(self, program: TrainingProgram
                            ) -> tuple[list[str], list[str]]:
        """Fallback: infer courses from catalog matching."""
        matched = self.catalog.for_program(program.name, limit=80)
        required_ids = []
        elective_ids = []
        for c in matched:
            if any("必修" in str(mp) or "核心" in str(mp) for mp in c.minor_programs):
                required_ids.append(c.id)
            else:
                elective_ids.append(c.id)
        return required_ids, elective_ids

    def _build_graph_courses(self, required_ids: list[str],
                             elective_ids: list[str],
                             program: TrainingProgram
                             ) -> list[GraphCourse]:
        """Build GraphCourse objects for the scheduler.

        Uses catalog data first; falls back to extracting name/credits
        from the program raw_text when a course is not in the catalog.
        """
        courses: list[GraphCourse] = []
        raw_text_cache: dict[str, tuple[str, float, str]] = {}

        def _lookup(cid: str) -> tuple[str, float, str, str]:
            """Return (name, credits, semester, prereqs) for a course ID."""
            # 1) Catalog
            rec = self._course_index.get(cid)
            if rec and rec.name and rec.name != cid:
                return (rec.name, rec.credits or 0, rec.semester or "", rec.prerequisites or "")

            # 2) Extract from raw_text (cached)
            if cid in raw_text_cache:
                name, cr, sem = raw_text_cache[cid]
                return (name, cr, sem, "")

            name, cr, sem = self._extract_course_from_text(cid, program.raw_text)
            raw_text_cache[cid] = (name, cr, sem)
            return (name, cr, sem, "")

        for cid in required_ids:
            name, credits, semester, prereqs = _lookup(cid)
            courses.append(GraphCourse(
                id=cid, name=name, credits=credits,
                semester=semester, raw_prereqs=prereqs,
                is_required=True, course_type="必修"
            ))

        for cid in elective_ids:
            name, credits, semester, prereqs = _lookup(cid)
            courses.append(GraphCourse(
                id=cid, name=name, credits=credits,
                semester=semester, raw_prereqs=prereqs,
                is_required=False, course_type="选修"
            ))

        return courses

    @staticmethod
    def _extract_course_from_text(cid: str, raw_text: str
                                  ) -> tuple[str, float, str]:
        """Extract course name, credits, and semester from raw_text.

        In the PDF text, courses appear as multi-line records:
            [course ID line]    10421263
            [course name line]  微积分C(1)
            [credits line]      3
            [semester+ lines]   秋 / 必修 / ...
        """
        idx = raw_text.find(cid)
        if idx < 0:
            return (cid, 0, "")

        # Get subsequent lines
        rest = raw_text[idx + len(cid):].strip()
        lines = rest.split("\n")
        name = cid
        credits = 0.0
        semester = ""

        if lines:
            # Line 1: course name
            candidate = lines[0].strip()
            if candidate and not candidate.isdigit() and len(candidate) < 80:
                name = candidate
            # Line 2: credits
            if len(lines) > 1:
                try:
                    credits = float(lines[1].strip())
                except ValueError:
                    pass
            # Lines 3+: semester hints
            for l in lines[2:6]:
                s = l.strip()
                if any(kw in s for kw in ("秋", "春", "夏", "春秋")):
                    if not semester:
                        semester = s[:10]
                    break

        return (name, credits, semester)

    @staticmethod
    def _semester_to_offset(sem: str) -> int:
        """Convert semester name to cycle offset."""
        for i, s in enumerate(SEMESTER_CYCLE):
            if s == sem:
                return i
        return 0  # default to 秋

    def _constrained_topological_sort(self, courses: list[GraphCourse],
                                      start_semester_offset: int = 0,
                                      total_semesters: int = 8
                                      ) -> list[list[GraphCourse]]:
        """Topological sort with semester constraints and credit awareness.

        Returns list of semesters (each a list of courses).
        """
        adj, course_map = build_prerequisite_graph(courses)

        # In-degree calculation
        in_degree: dict[str, int] = {}
        for name in adj:
            in_degree[name] = 0
        for name, prereqs in adj.items():
            for prereq in prereqs:
                if prereq in in_degree:
                    in_degree[name] += 1

        # Seed queue
        queue = deque()
        for name, deg in in_degree.items():
            if deg == 0 and name in course_map:
                queue.append(name)

        plan: list[list[GraphCourse]] = []
        taken: set[str] = set()
        all_names: set[str] = set(course_map.keys())

        for sem_idx in range(total_semesters):
            target_sem = SEMESTER_CYCLE[(start_semester_offset + sem_idx) % 3]

            # Refill queue
            if not queue:
                newly_ready = [
                    n for n in all_names
                    if n not in taken and in_degree.get(n, 0) == 0
                ]
                for n in newly_ready:
                    queue.append(n)

            semester_courses: list[GraphCourse] = []
            deferred: deque[str] = deque()

            while queue:
                name = queue.popleft()
                if name in taken or name not in course_map:
                    continue

                course = course_map[name]
                if not _course_offered_in_semester(course, target_sem):
                    deferred.append(name)
                    continue

                semester_courses.append(course)
                taken.add(name)

                # Reduce in-degree of dependents
                for other_name, prereqs in adj.items():
                    if name in prereqs and other_name in in_degree:
                        in_degree[other_name] -= 1
                        if in_degree[other_name] == 0 and other_name not in taken:
                            queue.append(other_name)

            # Push deferred back
            while deferred:
                n = deferred.popleft()
                if n not in taken:
                    queue.append(n)

            plan.append(semester_courses)

            if len(taken) >= len(all_names):
                break

        # Add unscheduled courses as a final warning semester
        unscheduled = [course_map[n] for n in all_names if n not in taken]
        if unscheduled:
            plan.append(unscheduled)

        return plan

    def _apply_credit_limits(self, raw_semesters: list[list[GraphCourse]],
                             grade_num: int, start_sem_offset: int
                             ) -> tuple[list[list[ScheduledCourse]], list[str]]:
        """Apply credit limits per semester and convert to ScheduledCourse."""
        warnings: list[str] = []
        result: list[list[ScheduledCourse]] = []
        max_cr = self.constraints.max_credits_per_semester
        hard_max = self.constraints.hard_max_credits

        for sem_idx, courses in enumerate(raw_semesters):
            scheduled: list[ScheduledCourse] = []
            total_credits = 0.0
            deferred: list[ScheduledCourse] = []

            for gc in courses:
                sc = ScheduledCourse(
                    course_id=gc.id,
                    course_name=gc.name,
                    credits=gc.credits,
                    semester_index=sem_idx,
                    semester_label=_semester_label(
                        grade_num + (start_sem_offset + sem_idx) // 3,
                        SEMESTER_CYCLE[(start_sem_offset + sem_idx) % 3]
                    ),
                    is_required=gc.is_required,
                    reason="必修课" if gc.is_required else "选修课",
                )

                if total_credits + gc.credits > hard_max:
                    deferred.append(sc)
                    warnings.append(
                        f"{sc.semester_label} 学分已达硬上限，"
                        f"{gc.name}({gc.credits}学分) 建议后续学期修读"
                    )
                    continue

                # Time conflict check — try each option, pick first non-conflicting
                chosen_opt, time_conflicts = self._find_non_conflicting_option(
                    gc.id, scheduled
                )
                if time_conflicts:
                    for conflict_msg in time_conflicts:
                        warnings.append(conflict_msg)

                if chosen_opt:
                    sc.chosen_option = chosen_opt
                    sc.time_slots = chosen_opt.slots
                    sc.reason += f" (选中{chosen_opt.class_id})" if chosen_opt.class_id else ""
                elif self._get_course_options(gc.id):
                    # Has time data but all options conflict → defer or force
                    if gc.is_required:
                        # Required course → force with first option, warn heavily
                        opts = self._get_course_options(gc.id)
                        sc.chosen_option = opts[0]
                        sc.time_slots = opts[0].slots
                        sc.reason += f" (所有班次均冲突，强制排入{opts[0].class_id}，请手动调整)"
                        scheduled.append(sc)
                        total_credits += gc.credits
                    else:
                        deferred.append(sc)
                    continue

                if gc.is_required or total_credits + gc.credits <= max_cr:
                    scheduled.append(sc)
                    total_credits += gc.credits
                else:
                    deferred.append(sc)

            if scheduled:
                result.append(scheduled)
            if deferred:
                # Try to fit deferred into next semester or add a warning semester
                result.append(deferred)

        return result, warnings

    def _apply_baoyan_constraints(self, semesters: list[list[ScheduledCourse]],
                                  student: StudentProfile, grade_num: int
                                  ) -> list[str]:
        """Apply 保研-specific constraints.

        Rules:
        - 保研核心课程必须在大四秋(第7学期)前完成
        - 标注影响GPA的关键课程并提醒
        - 建议前6学期保持高GPA
        """
        warnings: list[str] = []
        target_semester = self.constraints.baoyan_require_core_by_semester
        late_required: list[str] = []

        all_courses: list[ScheduledCourse] = []
        for sem in semesters:
            all_courses.extend(sem)

        for sc in all_courses:
            if sc.is_required and sc.semester_index >= target_semester:
                late_required.append(sc.course_name)

        if late_required:
            warnings.append(
                f"[!] 保研提醒：以下必修课安排在第{target_semester+1}学期或之后"
                f"（保研申请通常在大四秋），建议提前修读：{'、'.join(late_required[:5])}"
            )

        # GPA提醒
        if student.gpa > 0 and student.gpa < 3.0:
            warnings.append("[!] 当前GPA偏低，保研通常要求3.0以上，请重点关注本学期课程成绩。")
        elif student.gpa > 0 and student.gpa < 3.5:
            warnings.append("[i] GPA建议：保研竞争激烈，建议将GPA提升至3.5以上。")

        return warnings

    @staticmethod
    def _build_summary(semesters: list[list[ScheduledCourse]],
                       student: StudentProfile, program: TrainingProgram,
                       warnings: list[str]) -> str:
        """Build a human-readable summary."""
        total_required = 0.0
        total_elective = 0.0
        course_count = 0
        for sem in semesters:
            for sc in sem:
                course_count += 1
                if sc.is_required:
                    total_required += sc.credits
                else:
                    total_elective += sc.credits

        lines = [
            "═══════════════════════════════════",
            f"  培养方案：{program.name}",
            f"  学生专业：{student.major}　年级：{student.grade}",
            f"  总学分规划：必修{total_required:.0f} + 选修{total_elective:.0f} = {total_required + total_elective:.0f}学分",
            f"  课程总数：{course_count}门　规划学期：{len(semesters)}个",
        ]
        if student.goals:
            lines.append(f"  目标：{'、'.join(student.goals)}")
        if warnings:
            lines.append("  ────────────────────────────────")
            for w in warnings:
                lines.append(f"  {w}")
        lines.append("═══════════════════════════════════")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
# Formatter — pretty table output
# ═══════════════════════════════════════════════════════════════════════

def format_schedule(result: ScheduleResult) -> str:
    """Format a ScheduleResult as a pretty markdown table."""
    lines = [result.summary, ""]

    for sem_idx, semester in enumerate(result.semesters):
        if not semester:
            continue
        label = semester[0].semester_label
        total = sum(sc.credits for sc in semester)
        req_count = sum(1 for sc in semester if sc.is_required)
        ele_count = sum(1 for sc in semester if not sc.is_required)

        lines.append(f"## {label}")
        lines.append(f"> 必修{req_count}门 + 选修{ele_count}门　共{total:.0f}学分")
        lines.append("")
        lines.append("| 课程号 | 课程名称 | 学分 | 类型 | 说明 |")
        lines.append("|--------|----------|------|------|------|")
        for sc in semester:
            type_tag = "[必修]" if sc.is_required else "[选修]"
            lines.append(
                f"| {sc.course_id} | {sc.course_name} | {sc.credits} | {type_tag} | {sc.reason} |"
            )
        lines.append("")

    if result.warnings:
        lines.append("---")
        lines.append("### [!] 注意事项")
        for w in result.warnings:
            lines.append(f"- {w}")

    # 保研专项提示
    if "保研" in result.student.goals:
        lines.append("")
        lines.append("### [保研] 保研专项提示")
        lines.append("- 大四秋季学期（9月）是保研申请关键期，此前需完成所有必修课并取得较好成绩")
        lines.append("- 英语六级（CET-6）建议425分以上，部分院系有更高要求")
        lines.append("- 科研/竞赛经历是保研加分项，建议大二大三积极参与")
        lines.append("- 联系导师通常在大三下-大四秋，提前准备个人陈述和简历")

    return "\n".join(lines)


def generate_schedule(
    major: str, grade: str, program_name: str,
    completed_courses: str = "",
    gpa: float = 0.0,
    goals: str = "",
    target_semester: str = "",
    catalog: CourseCatalog = None,
) -> str:
    """Convenience function: generate a course schedule from string inputs.

    Args:
        major: 专业名
        grade: 年级 (大一/大二/大三/大四)
        program_name: 培养方案名
        completed_courses: 已修课程，逗号分隔 (如 "微积分,线性代数,10421263")
        gpa: 当前GPA (0表示未提供)
        goals: 目标，逗号分隔 (如 "保研,出国")
        target_semester: 开始排课的学期 (秋/春/夏)，默认为当前年级的秋季
        catalog: CourseCatalog实例

    Returns formatted schedule string.
    """
    from .data_loader import load_programs, get_program_by_name

    cat = catalog or CourseCatalog.load()
    programs = load_programs()
    program = get_program_by_name(program_name, programs)
    if not program:
        return f"未找到培养方案: {program_name}"

    # Parse completed courses
    completed = [c.strip() for c in completed_courses.replace("，", ",").split(",") if c.strip()]

    # Parse goals
    goal_list = [g.strip() for g in goals.replace("，", ",").split(",") if g.strip()]

    # Determine target semester
    if not target_semester:
        target_semester = "秋"  # default to fall

    student = StudentProfile(
        major=major,
        grade=grade,
        completed_courses=completed,
        gpa=gpa,
        goals=goal_list,
        target_semester_start=target_semester,
    )

    scheduler = CourseScheduler(cat)
    result = scheduler.generate(student, program)

    return format_schedule(result)
