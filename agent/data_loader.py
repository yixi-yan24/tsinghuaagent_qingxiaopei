import re, json, os
from dataclasses import dataclass, field, asdict
from typing import Optional

MARKDOWN_PATH = os.path.join(os.path.dirname(__file__), "..", "2025级培养方案.md")
CACHE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "programs.json")


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


def _clean_md_line(s: str) -> str:
    """Strip markdown heading markers (e.g. '## ') from a line."""
    return re.sub(r"^#+\s*", "", s.strip())


def parse_all() -> list[TrainingProgram]:
    """Parse the markdown training plan document into structured TrainingProgram objects.

    The markdown file is converted from the PDF and preserves the original structure:
    - Department headers (学院/系/书院/部)
    - Program titles (XX专业本科培养方案)
    - Program body starts with 一、培养目标
    """
    with open(MARKDOWN_PATH, encoding="utf-8") as f:
        text = f.read()

    lines = text.split("\n")

    # Find all section starts: lines that are department headers or program headers
    # A program section is: [department line] -> [program title line] -> [body starting with 一、培养目标]
    programs: list[TrainingProgram] = []

    # Strategy: find each "一、培养目标" and look backwards for title + department
    prog_starts = []
    for i, line in enumerate(lines):
        stripped = _clean_md_line(line)
        # Match program body start markers
        if re.match(r"一[、,]\s*培养目标", stripped):
            prog_starts.append(i)

    for start_idx in prog_starts:
        # Look backwards for program title and department
        title = ""
        department = ""
        header_start = start_idx

        for j in range(start_idx - 1, max(start_idx - 30, 0), -1):
            s = _clean_md_line(lines[j])
            if not s:
                continue
            # Skip page numbers and decorations
            if re.match(r"^\d{1,3}$", s) or s == "清华大学本科培养方案":
                continue
            # Detect program title: contains 专业 and 培养方案
            if not title and ("专业" in s or "培养方案" in s or "双学位" in s or "书院" in s):
                title = s
                header_start = j
                continue
            # Detect department: contains 学院/系/书院/部/院
            if not department and re.search(r"(学院|学系|系|书院|部|院)$", s):
                department = s
                header_start = j
                break

        if not title:
            # Fallback: use the line right before 一、培养目标
            for j in range(start_idx - 1, max(start_idx - 5, 0), -1):
                s = _clean_md_line(lines[j])
                if s and len(s) > 3:
                    title = s
                    header_start = j
                    break

        # Collect raw_text: from header_start to the next program's header_start
        # Find the next 一、培养目标 after this one
        next_start = None
        for ns in prog_starts:
            if ns > start_idx:
                next_start = ns
                break

        # Extract raw_text from header to next program (or end)
        if next_start:
            raw_lines = lines[header_start:next_start]
        else:
            raw_lines = lines[header_start:]

        raw_text = "\n".join(raw_lines).strip()

        prog = TrainingProgram(
            name=title or "未知专业",
            department=department or "未知院系",
            raw_text=raw_text,
        )
        _extract_metadata(prog)
        programs.append(prog)

    return programs


def _extract_metadata(prog: TrainingProgram):
    """Extract structured metadata from raw_text using regex."""
    text = prog.raw_text

    # Total credits
    m = re.search(r"(?:基本学分|总学分|学分要求).*?(\d[\d.]*)\s*学分", text)
    if not m:
        m = re.search(r"(\d[\d.]*)\s*学分", text[:400])
    if m:
        prog.total_credits = m.group(1) + "学分"

    # Degree
    m = re.search(r"(?:授予|学位).*?(\S+学[士硕博][士位])", text)
    if not m:
        m = re.search(r"(\S+学士学位)", text)
    if m:
        prog.degree = m.group(1).strip()

    # Duration (学制)
    m = re.search(r"学制[：:]\s*(.+?)(?=\n|$)", text)
    if m:
        prog.duration = m.group(1).strip()[:100]
    elif re.search(r"(\d年)", text[:500]):
        m2 = re.search(r"(\d年)", text[:500])
        if m2:
            prog.duration = m2.group(1).strip()

    # Prerequisites (先修课程)
    m = re.search(r"先修课程[要求]*[：:](.*?)(?=\n##|\n\d[、,.)]|\n[一二三四五][、,)]|\n[（(][一二三四五])", text, re.DOTALL)
    if m:
        prog.prerequisites = m.group(1).strip()[:500]

    # Contact phone
    m = re.search(r"(?:咨询)?电话[：:]\s*([\d-]+)", text)
    if m:
        prog.contact = m.group(1).strip()


def load_programs(force_reload: bool = False) -> list[TrainingProgram]:
    """Load programs from cache or parse from scratch."""
    if not force_reload and os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH, encoding="utf-8") as f:
                data = json.load(f)
            return [TrainingProgram(**item) for item in data]
        except Exception:
            pass

    programs = parse_all()
    try:
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump([asdict(m) for m in programs], f, ensure_ascii=False, indent=2)
    except PermissionError:
        pass
    return programs


def get_program_by_name(name: str, programs: list[TrainingProgram]) -> Optional[TrainingProgram]:
    """Find a training program by name or department, with graded fallback.

    Resolution order:
      1. name exact/substring match
      2. department exact/substring match
      3. keyword-in-name match
      4. keyword-in-department match
      5. semantic search (embedding) — lazy-loaded, only if needed
    """
    name = name.strip()
    if not name:
        return None

    # 1 — name match
    for m in programs:
        if name in m.name or m.name in name:
            return m
    # 2 — department match
    for m in programs:
        if name in m.department or m.department in name:
            return m
    # 3 — keyword in name
    for m in programs:
        for kw in name.split():
            if kw in m.name:
                return m
    # 4 — keyword in department
    for m in programs:
        for kw in name.split():
            if kw in m.department:
                return m

    # 5 — semantic search (lazy, so no impact on startup)
    try:
        from .embedding import semantic_search as _semantic_search
        results = _semantic_search(name, programs, top_k=1)
        if results and results[0][1] >= 0.5:
            return results[0][0]
    except Exception:
        pass

    return None


def search_programs(query: str, programs: list[TrainingProgram]) -> list[TrainingProgram]:
    """Search programs by keyword in name, department, or raw_text."""
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


def get_all_program_names(programs: list[TrainingProgram]) -> list[str]:
    return [m.name for m in programs]
