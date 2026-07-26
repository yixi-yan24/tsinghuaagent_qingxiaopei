import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Optional


COURSES_PATH = os.path.join(os.path.dirname(__file__), "..", "curated_courses.json")


def _normalize(value: str) -> str:
    return re.sub(r"[\s（）()\[\]【】,，.。:：;；/\\_-]+", "", value).lower()


@dataclass
class CourseRecord:
    id: str
    name: str
    department: str = ""
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
    minor_programs: list[dict[str, Any]] = field(default_factory=list)
    # 开课学期（待后续填充结构化数据）
    semester: str = ""          # "秋" / "春" / "夏" / "春秋"
    grade_level: str = ""       # "大一" / "大二" / ... (从原始数据推断)
    course_type: str = ""       # "必修" / "限选" / "选修" / "通识"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CourseRecord":
        return cls(
            id=str(data.get("id", "")).strip(),
            name=str(data.get("name", "")).strip(),
            department=str(data.get("department", "")).strip(),
            credits=data.get("credits"),
            total_hours=data.get("total_hours"),
            prerequisites=str(data.get("prerequisites", "")).strip(),
            description=str(data.get("description", "")).strip(),
            objectives=str(data.get("objectives", "")).strip(),
            expected_outcomes=str(data.get("expected_outcomes", "")).strip(),
            assessment_method=str(data.get("assessment_method", "")).strip(),
            grade_breakdown=str(data.get("grade_breakdown", "")).strip(),
            textbooks=str(data.get("textbooks", "")).strip(),
            instructor=str(data.get("instructor", "")).strip(),
            minor_programs=list(data.get("minor_programs") or []),
            semester=str(data.get("semester", "")).strip(),
            grade_level=str(data.get("grade_level", "")).strip(),
            course_type=str(data.get("course_type", "")).strip(),
        )


class CourseCatalog:
    """Searchable local catalog built from the curated course dataset."""

    def __init__(self, courses: list[CourseRecord]):
        self.courses = [course for course in courses if course.id and course.name]
        self._by_id = {course.id: course for course in self.courses}

    @classmethod
    def load(cls, path: str = COURSES_PATH) -> "CourseCatalog":
        try:
            with open(path, encoding="utf-8") as file:
                data = json.load(file)
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"无法加载课程目录: {path}") from exc
        if not isinstance(data, list):
            raise RuntimeError("课程目录格式错误：顶层必须是列表")
        return cls([CourseRecord.from_dict(item) for item in data if isinstance(item, dict)])

    def find(self, identifier: str) -> Optional[CourseRecord]:
        query = _normalize(identifier)
        if not query:
            return None
        if identifier.strip() in self._by_id:
            return self._by_id[identifier.strip()]
        exact = [course for course in self.courses if _normalize(course.name) == query]
        if len(exact) == 1:
            return exact[0]
        partial = [course for course in self.courses if query in _normalize(course.name)]
        return partial[0] if len(partial) == 1 else None

    def search(self, query: str, limit: int = 5) -> list[CourseRecord]:
        normalized_query = _normalize(query)
        if not normalized_query:
            return []

        scored: list[tuple[int, CourseRecord]] = []
        for course in self.courses:
            name = _normalize(course.name)
            program_names = " ".join(
                str(program.get("program", "")) for program in course.minor_programs
            )
            metadata = _normalize(f"{course.department} {program_names}")
            content = _normalize(
                f"{course.description[:2000]} {course.objectives[:1000]} "
                f"{course.expected_outcomes[:1000]} {course.prerequisites}"
            )

            score = 0
            if course.id == query.strip():
                score = 100
            elif name == normalized_query:
                score = 95
            elif normalized_query in name:
                score = 80
            elif normalized_query in metadata:
                score = 55
            elif normalized_query in content:
                score = 35
            if score:
                scored.append((score, course))

        scored.sort(key=lambda item: (-item[0], item[1].name, item[1].id))
        return [course for _, course in scored[:limit]]

    def for_program(self, program_name: str, limit: int = 40) -> list[CourseRecord]:
        """Find courses associated with a training program.

        Matches by: (1) exact minor_programs link, (2) department overlap,
        (3) keyword match in course metadata.
        """
        norm_name = _normalize(program_name.replace("专业培养方案", "").replace("培养方案", "").replace("本科", ""))
        if not norm_name:
            return []

        # Extract key terms from program name for fuzzy matching
        key_terms = [t for t in norm_name[:6] if len(t) > 1] if len(norm_name) <= 6 else [norm_name[:4]]

        matches: list[tuple[int, CourseRecord]] = []
        for course in self.courses:
            score = 0
            # 1) Exact program link
            for mp in (course.minor_programs or []):
                mp_name = _normalize(str(mp.get("program", "")))
                if norm_name in mp_name or mp_name in norm_name:
                    score = 100
                    break

            # 2) Department match
            if not score and course.department:
                dept_norm = _normalize(course.department)
                if norm_name and dept_norm and (norm_name[:4] in dept_norm or dept_norm[:4] in norm_name):
                    score = 70

            # 3) Keyword overlap with course metadata
            if not score:
                metadata = _normalize(f"{course.name} {course.description[:500]}")
                term_matches = sum(1 for t in key_terms if t in metadata)
                if term_matches >= 2:
                    score = 50
                elif term_matches >= 1:
                    score = 30

            if score:
                matches.append((score, course))

        matches.sort(key=lambda x: (-x[0], x[1].name))
        return [c for _, c in matches[:limit]]

    def recommend(self, major: str = "", grade: str = "", interests: str = "",
                  target_semester: str = "", limit: int = 10) -> list[tuple[int, CourseRecord]]:
        """Recommend courses based on student profile.

        Scoring dimensions (each 0-25):
        - Department match (major → department)
        - Interest keyword match in course name/description
        - Grade level appropriateness
        - Semester availability (待后续数据填充)
        - Course type relevance (通识优先 for 大一, 专业优先 for 大三+)

        Returns list of (score, CourseRecord) sorted by score descending.
        """
        interest_keywords = [kw.strip() for kw in interests.replace("、", ",").replace("，", ",").split(",") if kw.strip()] if interests else []
        grade_num = {"大一": 1, "大二": 2, "大三": 3, "大四": 4, "大五": 5}.get(grade, 0)

        scored: list[tuple[int, CourseRecord]] = []
        for course in self.courses:
            score = 0

            # 1) Department match (0-25)
            dept_norm = _normalize(course.department or "")
            major_norm = _normalize(major)
            if major_norm and dept_norm:
                if major_norm[:4] in dept_norm or dept_norm[:4] in major_norm:
                    score += 25
                elif any(t in dept_norm for t in major_norm[:4]):
                    score += 15
                else:
                    score += 5   # still show other dept courses

            # 2) Interest match (0-25)
            if interest_keywords:
                course_text = _normalize(
                    f"{course.name} {course.description[:500]} {course.objectives[:300]}"
                )
                hits = sum(1 for kw in interest_keywords if _normalize(kw) in course_text)
                if hits >= 3:
                    score += 25
                elif hits >= 2:
                    score += 20
                elif hits >= 1:
                    score += 15

            # 3) Grade level (0-25) — heuristic based on course content
            # Lower-grade courses: more foundational terms
            course_text = _normalize(f"{course.name} {course.description[:300]}")
            foundation_words = ["基础", "概论", "导论", "入门", "初探", "基本", "原理"]
            advanced_words = ["高级", "前沿", "研究", "专题", "研讨", "实践", "设计"]
            foundation_score = sum(1 for w in foundation_words if w in course_text)
            advanced_score = sum(1 for w in advanced_words if w in course_text)

            if grade_num <= 1:  # 大一 → prefer foundation
                score += min(25, foundation_score * 8 + 5)
            elif grade_num <= 2:  # 大二 → balanced
                score += min(25, foundation_score * 5 + advanced_score * 5 + 5)
            elif grade_num >= 3:  # 大三+ → prefer advanced
                score += min(25, advanced_score * 8 + 5)

            # 4) Semester availability (0-25) — placeholder, always 15 for now
            # TODO: use course.semester when data is populated
            if target_semester:
                if course.semester and target_semester in course.semester:
                    score += 25
                elif not course.semester:
                    score += 15  # unknown → neutral
            else:
                score += 15

            if score > 0:
                scored.append((score, course))

        scored.sort(key=lambda x: (-x[0], x[1].name))
        return scored[:limit]
