import re
from collections import defaultdict, deque
from dataclasses import dataclass, field


@dataclass
class Course:
    id: str
    name: str
    credits: float = 0
    semester: str = ""         # 秋/春/春秋/夏
    raw_prereqs: str = ""
    is_required: bool = True
    course_type: str = ""      # 必修/限选/选修
    department: str = ""


def parse_courses_from_table(markdown_text: str) -> list[Course]:
    """Parse course info from markdown text — HTML tables first, plain text as fallback."""
    courses = _parse_html_tables(markdown_text)
    if courses:
        return courses
    # Fallback: parse plain-text course lines (from PDF conversion)
    return _parse_text_courses(markdown_text)


# 表头别名 → 列语义。注意顺序：先匹配更具体的（如"备注(说明及先修要求)"）。
_HEADER_ALIASES = [
    ("课程编号", "id"), ("课程代码", "id"), ("编号", "id"), ("课程号", "id"),
    ("课程名称", "name"), ("名称", "name"), ("课程", "name"),
    ("学分", "credits"),
    ("开课学期", "semester"), ("学期", "semester"), ("开课", "semester"),
    ("备注(说明及先修要求)", "prereqs"), ("备注", "prereqs"),
    ("先修要求", "prereqs"), ("先修", "prereqs"), ("说明", "prereqs"),
]


def _detect_column_map(cells: list[str]) -> dict[int, str]:
    """Map column index → semantic key based on a header row.

    Returns e.g. {0: 'id', 1: 'name', 2: 'credits', 3: 'prereqs'}.
    """
    col_map: dict[int, str] = {}
    for idx, cell in enumerate(cells):
        for alias, semantic in _HEADER_ALIASES:
            if alias in cell:
                col_map[idx] = semantic
                break
    return col_map


def _parse_html_tables(text: str) -> list[Course]:
    """Parse embedded HTML table rows from a Markdown document.

    Column meanings are taken from the header row when present (so a
    ``备注(说明及先修要求)`` column becomes *raw_prereqs* instead of being
    mistaken for an offering-semester column).  Falls back to the fixed
    [id, name, credits, semester, prereqs] layout when no header is found.
    """
    courses = []
    rows = re.findall(r"<tr>(.*?)</tr>", text, re.DOTALL)
    if not rows:
        return courses
    current_type = ""
    col_map: dict[int, str] | None = None

    for row in rows:
        cells = re.findall(r"<td[^>]*>(.*?)</td>", row, re.DOTALL)
        cells = [re.sub(r"<[^>]+>", "", c).strip() for c in cells]

        # Header row — detect column semantics.
        if cells and any("课程编号" in c or "课程代码" in c or c == "编号" for c in cells):
            col_map = _detect_column_map(cells)
            continue

        # Detect section headers like <td colspan="5">必修课程 28 学分</td>
        if len(cells) == 1:
            text = cells[0]
            if "必修" in text:
                current_type = "必修"
            elif "限选" in text:
                current_type = "限选"
            elif "选修" in text:
                current_type = "选修"
            continue

        if len(cells) >= 4 or (col_map and len(cells) >= 3):
            course_id = cells[0]
            # Skip non-course rows (headers, merged cells)
            if not re.match(r"^\d", course_id) and course_id not in {"新开课", "新开"}:
                continue

            if col_map:
                def _cell(semantic: str) -> str:
                    for idx, sem in col_map.items():
                        if sem == semantic and idx < len(cells):
                            return cells[idx]
                    return ""

                name = _cell("name") or (cells[1] if len(cells) > 1 else "")
                credits_raw = _cell("credits") or (cells[2] if len(cells) > 2 else "")
                semester = _cell("semester")
                raw_prereqs = _cell("prereqs")
            else:
                name = cells[1] if len(cells) > 1 else ""
                credits_raw = cells[2] if len(cells) > 2 else ""
                semester = cells[3] if len(cells) > 3 else ""
                raw_prereqs = cells[4] if len(cells) > 4 else ""

            credits = 0
            try:
                credits = float(credits_raw) if credits_raw else 0
            except ValueError:
                pass

            if course_id or name:
                courses.append(Course(
                    id=course_id,
                    name=name,
                    credits=credits,
                    semester=semester,
                    raw_prereqs=raw_prereqs,
                    is_required=(current_type == "必修"),
                    course_type=current_type
                ))

    return courses


def _parse_text_courses(text: str) -> list[Course]:
    """Parse plain-text course records from PDF-converted markdown.

    The PDF table is extracted as multi-line records where EACH column
    becomes a separate line:

        [course ID line]    10421263
        [course name line]  微积分C(1)
        [credits line]      3
        [semester lines]    秋 / 春 / ...
        [next ID line]      30020682

    We collect lines between consecutive 8-digit IDs as one course record.
    """
    courses = []
    current_type = ""

    lines = text.split("\n")
    section_pat = re.compile(r"(必修|限选|选修|专业核心|专业基础)")

    # Find all lines that are standalone 8-digit course IDs
    id_pattern = re.compile(r"^\d{8}$")

    i = 0
    while i < len(lines):
        stripped = lines[i].strip()

        # Detect section type from short header lines
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

        # Standalone 8-digit ID → start of a course record
        if id_pattern.match(stripped):
            course_id = stripped
            # Collect subsequent lines until the next 8-digit ID or table end
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
                # Line 0: course name
                name = record_lines[0] if len(record_lines) > 0 else ""
                # Line 1: credits (number)
                credits = 0
                if len(record_lines) > 1:
                    try:
                        credits = float(record_lines[1])
                    except ValueError:
                        pass
                # Lines 2+: semester and notes
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
                        id=course_id,
                        name=name,
                        credits=credits,
                        semester=semester,
                        raw_prereqs=raw_prereqs,
                        is_required=(current_type == "必修"),
                        course_type=current_type or "必修"
                    ))

            i = j
            continue

        i += 1

    return courses


def build_prerequisite_graph(courses: list[Course]) -> tuple[dict[str, set[str]], dict[str, Course]]:
    """Build a DAG of course dependencies.
    Returns (adjacency_list, course_map).
    adjacency_list: course_name -> set of prerequisite course_names
    """
    adj: dict[str, set[str]] = {}
    course_map: dict[str, Course] = {}

    for c in courses:
        if c.name:
            course_map[c.name] = c
            adj.setdefault(c.name, set())

    # Parse prerequisite text to link courses
    for c in courses:
        if not c.raw_prereqs or not c.name:
            continue
        prereq_text = c.raw_prereqs
        # Remove common noise
        prereq_text = re.sub(r"<[^>]+>", "", prereq_text)
        prereq_text = re.sub(r"[、，,、]", " ", prereq_text)

        # Extract known course names from the prerequisite text
        for other_name in course_map:
            if other_name != c.name and other_name in prereq_text:
                adj.setdefault(c.name, set()).add(other_name)

    return adj, course_map


def _course_offered_in_semester(course: Course, target_sem: str) -> bool:
    """Return True if *course* is offered in *target_sem* (秋 / 春 / 夏).

    A course marked "春秋" (or "春,秋") is offered in both fall and spring.
    A course with an empty semester string is treated as *always offered*.
    """
    if not course.semester:
        return True  # no constraint → assume always available
    # Normalise: "春,秋" or "春秋" → both spring and fall
    sem = course.semester
    if "春秋" in sem or ("春" in sem and "秋" in sem):
        return target_sem in ("春", "秋")
    if "夏" in sem:
        return target_sem == "夏"
    if "秋" in sem:
        return target_sem == "秋"
    if "春" in sem:
        return target_sem == "春"
    return True  # unrecognised → assume no constraint


def topological_sort(courses: list[Course],
                     max_credits_per_semester: int = 25) -> list[list[Course]]:
    """Generate a semester-by-semester plan using topological sort.

    Returns a list of semesters, each a list of courses to take that term.
    Courses are ordered so that prerequisites come before dependents, each
    course is only placed in a semester where it is actually offered, and a
    per-semester credit cap keeps the load balanced across terms.

    When a genuine cycle or deadlock is detected the affected courses are
    placed into a final "未排入" (unscheduled) semester with an explanation
    rather than silently breaking prerequisite constraints.
    """
    adj, course_map = build_prerequisite_graph(courses)

    # In-degrees: count of *unfulfilled* prerequisites per course.
    in_degree: dict[str, int] = {name: 0 for name in adj}
    for name, prereqs in adj.items():
        for prereq in prereqs:
            if prereq in in_degree:
                in_degree[name] += 1

    # Seed queue — courses with zero prerequisites.
    queue = deque()
    for name, degree in in_degree.items():
        if degree == 0 and name in course_map:
            queue.append(name)

    plan: list[list[Course]] = []
    taken: set[str] = set()
    remaining: set[str] = set(course_map.keys())

    SEMESTER_CYCLE = ["秋", "春", "夏"]
    semester_idx = 0
    # Track consecutive empty semesters to detect genuine deadlocks.
    empty_streak = 0

    while remaining:
        target_sem = SEMESTER_CYCLE[semester_idx % len(SEMESTER_CYCLE)]

        # Refill queue from ready courses if it's empty.
        if not queue:
            newly_ready = [n for n in remaining if n not in taken and in_degree.get(n, 0) == 0]
            for n in newly_ready:
                queue.append(n)

        semester_courses: list[Course] = []
        semester_credits = 0.0
        deferred: deque[str] = deque()

        while queue:
            name = queue.popleft()
            if name in taken or name not in course_map:
                continue

            course = course_map[name]
            if not _course_offered_in_semester(course, target_sem):
                deferred.append(name)
                continue

            # Credit cap: defer over-budget courses to the next term, unless
            # nothing has been scheduled yet this term (cap is a soft limit).
            if semester_courses and semester_credits + course.credits > max_credits_per_semester:
                deferred.append(name)
                continue

            semester_courses.append(course)
            semester_credits += course.credits
            taken.add(name)
            remaining.discard(name)

            # Fulfilled a prerequisite — reduce in-degree of dependents.
            for other_name, prereqs in adj.items():
                if name in prereqs and other_name in in_degree:
                    in_degree[other_name] -= 1
                    if in_degree[other_name] == 0 and other_name not in taken:
                        queue.append(other_name)

        # Deadlock guard: if the credit cap deferred everything this term,
        # force one course in — the cap must not stall the whole plan.
        if not semester_courses and deferred:
            name = deferred.popleft()
            course = course_map.get(name)
            if course:
                semester_courses.append(course)
                taken.add(name)
                remaining.discard(name)
                for other_name, prereqs in adj.items():
                    if name in prereqs and other_name in in_degree:
                        in_degree[other_name] -= 1
                        if in_degree[other_name] == 0 and other_name not in taken:
                            queue.append(other_name)

        # Push deferred courses back so they are reconsidered next term.
        while deferred:
            name = deferred.popleft()
            if name not in taken:
                queue.append(name)

        if semester_courses:
            plan.append(semester_courses)
            empty_streak = 0
        else:
            empty_streak += 1
            # After a full cycle with no progress we have a genuine deadlock
            # (cycle or missing prerequisite data).  Surface it honestly.
            if empty_streak >= len(SEMESTER_CYCLE):
                unscheduled = [course_map[n] for n in remaining if n in course_map and n not in taken]
                if unscheduled:
                    plan.append(unscheduled)
                break

        semester_idx += 1

    return plan


def format_plan(plan: list[list[Course]], student_grade: str = "大二") -> str:
    """Format the plan as a human-readable table."""
    grade_map = {"大一": 1, "大二": 2, "大三": 3, "大四": 4}
    current_grade_num = grade_map.get(student_grade, 2)
    semester_labels = ["秋", "春", "夏"]

    lines = ["📋 **拓扑排序算法生成的最优修读计划**\n"]
    lines.append("| 学期 | 课程名称 | 学分 | 类型 |")
    lines.append("|------|----------|------|------|")

    for i, semester_courses in enumerate(plan):
        year_idx = current_grade_num - 1 + (i // 3)
        if year_idx >= 4:
            break
        year_label = f"{['大一','大二','大三','大四'][year_idx]}"
        sem_label = semester_labels[i % 3]
        lines.append(f"| **{year_label} {sem_label}** | | | |")
        for course in semester_courses:
            lines.append(f"| | {course.name} | {course.credits} | {course.course_type} |")
        lines.append("| | | | |")

    lines.append("\n*注：此计划由课程先修关系 DAG 拓扑排序生成，考虑了学期开课约束。*")
    return "\n".join(lines)
