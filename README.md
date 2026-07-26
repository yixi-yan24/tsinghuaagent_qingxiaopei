# Tsinghua Training Plan Advisor — 清华大学培养方案助手

基于 DeepSeek API 构建的智能 Agent，为清华本科生提供培养方案咨询与课程规划服务。兼容 OpenAI `/v1/chat/completions` 格式，支持流式输出。

---

## 快速开始

### 前置要求

- Python 3.10+
- DeepSeek API Key（或兼容 OpenAI 格式的其他 API）

### 本地运行

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置 API Key（二选一）
export DEEPSEEK_API_KEY=sk-xxx          # 环境变量
# 或将 Key 写入 .config 文件（自动读取）

# 3. 启动 CLI 交互模式
python run.py

# 或启动 API 服务
python run.py --mode api --port 8000
```

### Docker 部署

```bash
# 构建镜像
docker build -t tsinghua-training-plan-advisor .

# 运行
docker run -d --name training-plan-advisor \
  -p 8000:8000 \
  -e DEEPSEEK_API_KEY=sk-xxx \
  -v ./data:/app/data \
  tsinghua-training-plan-advisor
```

或使用 docker compose：

```bash
export DEEPSEEK_API_KEY=sk-xxx
docker compose up -d
```

---

## API 文档

### OpenAI 兼容接口

**对话：** `POST /v1/chat/completions`

```json
{
  "model": "tsinghua-training-plan-advisor",
  "messages": [
    {"role": "user", "content": "我是计算机系大一学生，想了解本专业的培养方案"}
  ],
  "stream": true,
  "user": "会话ID（可选）"
}
```

可用任何 OpenAI SDK 调用：

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
resp = client.chat.completions.create(
    model="tsinghua-training-plan-advisor",
    messages=[{"role": "user", "content": "..."}],
    stream=True
)
for chunk in resp:
    print(chunk.choices[0].delta.content or "", end="")
```

**模型列表：** `GET /v1/models`

### 培养方案数据接口

| 端点 | 说明 |
|------|------|
| `GET /programs` | 获取所有培养方案列表 |
| `GET /programs/{名称}` | 获取某培养方案详细信息 |

---

## 项目架构

```
TsingXiaoPeiAgent/
├── agent/                  # Agent 核心
│   ├── core.py             # 会话管理、LLM 调用、ReAct 工具调度
│   ├── data_loader.py      # 长期记忆：解析培养方案 → 结构化数据
│   ├── embedding.py        # 词嵌入引擎：语义搜索（sentence-transformers）
│   ├── memory.py           # 短期记忆（对话历史）& 长期记忆（培养方案数据库）
│   ├── tools.py            # 工具集：搜索、详情、要求检查、排课、课程推荐
│   ├── course_catalog.py   # 已整理课程资料的本地检索目录
│   ├── course_graph.py     # 课程 DAG 拓扑排序算法
│   ├── scheduler.py        # 智能排课引擎：基于约束优化的课程表自动生成
│   ├── llm_client.py       # 统一的模型调用、超时与瞬时失败重试
│   ├── prompts.py          # 系统提示词模板
│   ├── planner.py          # 修读计划生成
│   └── multi_agent.py      # Multi-Agent 协同搜索与审核
├── api/
│   └── main.py             # FastAPI 服务（OpenAI 兼容格式）
├── data/                   # 解析缓存（自动生成）
├── Dockerfile & docker-compose.yml
├── run.py                  # 统一入口
├── curated_courses.json    # 课程介绍
└── 2025级培养方案.md        # 培养方案源文件
```

### 设计要点

| 组件 | 说明 |
|------|------|
| **推理机制** | ReAct 模式：LLM 输出 `ACTION` 触发工具调用，结果回填后二次推理 |
| **短期记忆** | 每个会话独立的对话历史（最近 20 轮），以 `user` 字段区分 |
| **长期记忆** | 本科专业培养方案结构化数据 + 1000+ 门已整理的课程资料 |
| **词嵌入** | 基于 `shibing624/text2vec-base-chinese` 的语义搜索，余弦相似度排序 |
| **规划能力** | LLM 自主推理 + 拓扑排序双通道，考虑先修关系、开课学期、学分均衡 |
| **排课引擎** | 约束优化自动排课：支持先修关系 DAG、开课学期、学分上限（25/学期）、时间冲突检测、保研约束、已修课程排除 |
| **工具集** | list_programs / search_programs / semantic_search / get_program_detail / check_requirements / multi_agent_search / search_courses / get_course_detail / list_program_courses / recommend_courses / generate_schedule |

---

## 使用示例

### CLI 模式

```
$ python run.py

清华大学培养方案助手 v1.0
输入 'quit' 退出 | 'clear' 清空对话 | 'plan' 生成修读计划

你 > 我是计算机系大二学生，想了解培养方案
助手 > [根据你的专业介绍培养方案，并给出课程安排建议...]

你 > plan 计算机科学与技术 大二 计算机科学与技术专业培养方案
助手 > [生成按学期的详细修读计划...]
```

### API 模式

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tsinghua-training-plan-advisor",
    "messages": [{"role": "user", "content": "计算机专业有哪些必修课？"}],
    "stream": true
  }'
```

---

## 配环境

依赖：`fastapi`、`uvicorn`、`httpx`、`pydantic`、`sentence-transformers`

```bash
pip install fastapi uvicorn httpx pydantic sentence-transformers
```

> 首次运行词嵌入功能时会自动下载模型（约 400MB），国内已配置 HF 镜像加速。也可通过环境变量 `HF_ENDPOINT` 自定义镜像源。

---

## License

MIT
