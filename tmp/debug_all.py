#!/usr/bin/env python3
"""Full project debug — checks syntax, imports, data, tools, scheduler, edge cases."""
import sys, io, os, re, traceback, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

errors = []
warnings = []

def check(desc, fn):
    try:
        fn()
        return True
    except Exception as e:
        errors.append((desc, str(e)[:200]))
        return False

# ═══════════ 1. Syntax ═══════════
print('=' * 60)
print('1. Syntax & Imports')
print('=' * 60)

import py_compile
files = [
    'run.py',
    'agent/__init__.py', 'agent/core.py', 'agent/tools.py',
    'agent/prompts.py', 'agent/data_loader.py', 'agent/memory.py',
    'agent/llm_client.py', 'agent/course_catalog.py', 'agent/course_graph.py',
    'agent/embedding.py', 'agent/planner.py', 'agent/multi_agent.py',
    'agent/scheduler.py',
    'api/__init__.py', 'api/main.py',
]
for fpath in files:
    try:
        py_compile.compile(fpath, doraise=True)
    except py_compile.PyCompileError as e:
        errors.append((f'Syntax: {fpath}', str(e)[:200]))
        print(f'  FAIL: {fpath}')
    else:
        pass  # silent OK
print(f'  Syntax: {len(files) - sum(1 for d,_ in errors if "Syntax" in d)}/{len(files)} OK')

# Module imports
try:
    from agent.llm_client import chat_completion, chat_completion_stream
    from agent.memory import ShortTermMemory, LongTermMemory
    from agent.course_graph import parse_courses_from_table, topological_sort, format_plan
    from agent.course_catalog import CourseCatalog, CourseRecord
    from agent.prompts import SYSTEM_PROMPT
    from agent.multi_agent import MultiAgentSystem, SpecialistAgent
    from agent.planner import CoursePlanner
    from agent.tools import Tools
    from agent.core import TrainingPlanAgent, AgentSession
    from agent.data_loader import load_programs, TrainingProgram, search_programs, get_all_program_names
    from agent.embedding import semantic_search
    from agent.scheduler import (
        generate_schedule, CourseScheduler, StudentProfile,
        ScheduleConstraints, TimeSlot, CourseOption
    )
    print('  Imports: OK')
except Exception as e:
    errors.append(('imports', str(e)))
    print(f'  Imports: FAILED - {e}')
    print('  CRITICAL — cannot continue')
    for desc, msg in errors:
        print(f'    [{desc}] {msg}')
    sys.exit(1)

# ═══════════ 2. Data ═══════════
print()
print('=' * 60)
print('2. Data Layer')
print('=' * 60)

if os.path.exists('data/programs.json'):
    os.remove('data/programs.json')

try:
    programs = load_programs()
    print(f'  Programs: {len(programs)}')
except Exception as e:
    errors.append(('load_programs', str(e)))
    print(f'  Programs: ERROR - {e}')
    programs = []

try:
    cat = CourseCatalog.load()
    print(f'  Catalog: {len(cat.courses)} courses')
except Exception as e:
    errors.append(('CourseCatalog', str(e)))
    print(f'  Catalog: ERROR - {e}')
    cat = None

# Data quality
if programs:
    names = get_all_program_names(programs)
    empty_name = sum(1 for n in names if not n or n == '未知专业')
    empty_dept = sum(1 for p in programs if not p.department or p.department == '未知院系')
    empty_credits = sum(1 for p in programs if not p.total_credits)
    short_text = sum(1 for p in programs if len(p.raw_text) < 100)
    print(f'  Quality: empty_name={empty_name} empty_dept={empty_dept} empty_credits={empty_credits} short_text={short_text}')
    if empty_name: warnings.append(f'{empty_name} programs have empty names')
    if empty_dept: warnings.append(f'{empty_dept} programs have empty departments')
    if empty_credits: warnings.append(f'{empty_credits} programs have no credit info')

    # Duplicate check
    from collections import Counter
    name_counts = Counter(names)
    dupes = [(n, c) for n, c in name_counts.items() if c > 1]
    if dupes:
        warnings.append(f'{len(dupes)} duplicate program names')
        print(f'  Duplicates: {len(dupes)} (e.g. {dupes[0][0][:30]} x{dupes[0][1]})')

# ═══════════ 3. Course Graph ═══════════
print()
print('=' * 60)
print('3. Course Graph')
print('=' * 60)

if programs:
    total_courses = 0
    with_courses = 0
    with_plan = 0
    for p in programs:
        courses = parse_courses_from_table(p.raw_text)
        if courses:
            with_courses += 1
            total_courses += len(courses)
            plan = topological_sort(courses)
            if plan:
                with_plan += 1
    print(f'  Parsed: {with_courses}/{len(programs)} programs, {total_courses} courses total')
    print(f'  Plans: {with_plan} programs have valid topological sort')

    # Sample
    sample_p = programs[0]
    sample_courses = parse_courses_from_table(sample_p.raw_text)
    if sample_courses:
        c = sample_courses[0]
        print(f'  Sample: {c.id} | {c.name[:30]} | {c.credits}cr | {c.course_type}')

# ═══════════ 4. Memory & Tools ═══════════
print()
print('=' * 60)
print('4. Memory & Tools')
print('=' * 60)

if programs and cat:
    ltm = LongTermMemory(programs)
    tools = Tools(ltm)

    # STM
    stm = ShortTermMemory()
    stm.add('system', 'test')
    stm.add('user', 'hello')
    stm.add('assistant', 'hi')
    fmt = stm.to_llm_format()
    print(f'  STM: {len(stm.messages)} msgs, LLM fmt: {len(fmt)} entries')
    if len(stm.messages) != 3:
        warnings.append('STM message count mismatch')

    # Tool tests
    tool_checks = [
        ('list_programs', lambda: len(tools.list_programs()) > 100),
        ('search_programs', lambda: isinstance(tools.search_programs('计算机'), str)),
        ('search_courses', lambda: isinstance(tools.search_courses('微积分'), str)),
        ('get_course_detail', lambda: '30240184' in tools.get_course_detail('30240184') or '未找到' in tools.get_course_detail('30240184')),
        ('get_program_detail', lambda: len(tools.get_program_detail(programs[0].name)) > 50),
        ('check_requirements', lambda: isinstance(tools.check_requirements('计算机', programs[0].name), str)),
        ('recommend_courses', lambda: isinstance(tools.recommend_courses(major='计算机', grade='大二'), str)),
    ]
    all_ok = True
    for name, fn in tool_checks:
        try:
            if not fn():
                warnings.append(f'Tool {name}: unexpected result')
                all_ok = False
        except Exception as e:
            errors.append((f'Tool {name}', str(e)[:200]))
            all_ok = False
    if all_ok:
        print(f'  Tools: all OK ({len(tool_checks)} tested)')
    else:
        print(f'  Tools: some issues found')

# ═══════════ 5. Scheduler ═══════════
print()
print('=' * 60)
print('5. Scheduler')
print('=' * 60)

if programs and cat:
    cs_list = search_programs('计算机科学', programs)
    prog = cs_list[0] if cs_list else programs[3]

    try:
        result = generate_schedule(
            major='计算机', grade='大二', program_name=prog.name,
            completed_courses='微积分,线性代数', gpa=3.5, goals='保研',
            target_semester='秋', catalog=cat,
        )
        scheduled = len(re.findall(r'\|\s*\d{8}\s+\|', result))
        named = sum(1 for m in re.finditer(r'\|\s*(\d{8})\s+\|(.+?)\s+\|', result)
                    if m.group(1) != m.group(2).strip()[:8])
        zero_cr = sum(1 for m in re.finditer(r'\|\s*(\d{8})\s+\|.+\|\s*0\s+\|', result))
        print(f'  Schedule: {scheduled} courses, {named} named, {zero_cr} zero-credit')
        if zero_cr > 0:
            warnings.append(f'{zero_cr} courses have 0 credits in schedule')
    except Exception as e:
        errors.append(('generate_schedule', str(e)[:200]))
        print(f'  Schedule: ERROR - {e}')

    # TimeSlot logic
    ts1 = TimeSlot(day=1, start_period=1, end_period=2)
    ts2 = TimeSlot(day=1, start_period=2, end_period=3)
    ts3 = TimeSlot(day=2, start_period=1, end_period=2)
    assert ts1.overlaps(ts2), 'overlap fail'
    assert not ts1.overlaps(ts3), 'day diff fail'
    print(f'  TimeSlot: OK')

    # CourseOption logic
    opt1 = CourseOption(class_id='A', slots=[TimeSlot(1,1,2)])
    opt2 = CourseOption(class_id='B', slots=[TimeSlot(1,1,2)])
    opt3 = CourseOption(class_id='C', slots=[TimeSlot(2,1,2)])
    assert opt1.conflicts_with(opt2), 'same slot should conflict'
    assert not opt1.conflicts_with(opt3), 'diff day should not conflict'
    print(f'  CourseOption: OK')

# ═══════════ 6. Edge Cases ═══════════
print()
print('=' * 60)
print('6. Edge Cases')
print('=' * 60)

if programs and cat:
    tests = [
        ('empty search', lambda: '提供' in tools.search_programs('') or '关键词' in tools.search_programs('')),
        ('nonexistent course', lambda: '未找到' in tools.get_course_detail('99999999')),
        ('nonexistent program', lambda: '未找到' in tools.get_program_detail('不存在的方案xyz')),
        ('partial name match', lambda: len(tools.get_program_detail('计算机')) > 50),
        ('recommend no input', lambda: '提供' in tools.recommend_courses() or '专业' in tools.recommend_courses()),
        ('missing program schedule', lambda: '未找到' in generate_schedule('计算机', '大一', '不存在的方案', catalog=cat)),
    ]
    ok = 0
    for desc, fn in tests:
        try:
            if fn():
                ok += 1
            else:
                warnings.append(f'Edge case: {desc} - unexpected result')
        except Exception as e:
            errors.append((f'Edge: {desc}', str(e)[:200]))
    print(f'  Edge cases: {ok}/{len(tests)} OK')

# ═══════════ 7. Agent Core ═══════════
print()
print('=' * 60)
print('7. Agent Core (no API calls)')
print('=' * 60)

try:
    agent = TrainingPlanAgent(api_key='sk-test')
    session = agent.create_session()
    print(f'  Session: STM={len(session.stm.messages)} msgs, tool_block_cached={session._tool_block is None}')
    print(f'  Dispatch table: {len(session._TOOL_PARAM_MAP)} tools')

    # Planner
    planner = CoursePlanner(api_key='sk-test')
    if programs:
        plan_result = planner.generate_plan('计算机', '大二', programs[3].name, ltm)
        print(f'  Planner: {len(plan_result)} chars output')

    # Multi-agent
    mas = MultiAgentSystem(api_key='sk-test')
    profile = mas.analyze_profile([{'role': 'user', 'content': '计算机系大二 对AI感兴趣'}])
    print(f'  Multi-agent: profile={profile}')
except Exception as e:
    errors.append(('Agent core', str(e)[:200]))
    print(f'  Agent core: ERROR - {e}')

# ═══════════ FINAL ═══════════
print()
print('=' * 60)
print('FINAL REPORT')
print('=' * 60)

if errors:
    print(f'  ERRORS ({len(errors)}):')
    for desc, msg in errors:
        print(f'    [{desc}] {msg[:150]}')
else:
    print(f'  Errors: 0')

if warnings:
    print(f'  WARNINGS ({len(warnings)}):')
    for w in warnings:
        print(f'    {w[:150]}')
else:
    print(f'  Warnings: 0')

print()
if errors:
    print('DEBUG FAILED — fix errors above')
    sys.exit(1)
else:
    print('DEBUG PASSED — all systems operational')
    sys.exit(0)
