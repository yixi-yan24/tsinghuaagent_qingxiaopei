import os, sys, time, uuid, json, logging
from collections import OrderedDict
from threading import Lock
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, Literal
from agent.core import TrainingPlanAgent
from agent.data_loader import load_programs
from agent.core import AgentSession

def _get_api_key() -> str:
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if key and key.startswith("sk-"):
        return key
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".config")
    if os.path.isfile(config_path):
        with open(config_path, encoding="utf-8") as f:
            key = f.read().strip()
    if not key or not key.startswith("sk-"):
        raise RuntimeError("请通过环境变量 DEEPSEEK_API_KEY 或 .config 文件设置有效的 DeepSeek API Key")
    return key

API_KEY = _get_api_key()

_agent = TrainingPlanAgent(api_key=API_KEY)
_programs = load_programs()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Tsinghua Training Plan Advisor API (OpenAI-compatible)",
    description="清华大学培养方案助手 — 兼容 OpenAI /v1/chat/completions 格式",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# === OpenAI-compatible Schemas ===

class ChatCompletionMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str = Field(min_length=1, max_length=12_000)

class ChatCompletionRequest(BaseModel):
    model: str = Field(default="tsinghua-training-plan-advisor", description="模型名，本服务固定为此值")
    messages: list[ChatCompletionMessage] = Field(min_length=1, max_length=50, description="对话消息列表")
    temperature: float = Field(default=0.3, ge=0, le=2)
    max_tokens: int = Field(default=4096, ge=1, le=8192)
    stream: bool = Field(default=False, description="是否流式返回（SSE 格式）")
    user: Optional[str] = Field(default=None, max_length=128, description="会话标识，用于保持上下文")

class Choice(BaseModel):
    index: int
    message: ChatCompletionMessage
    finish_reason: str

class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: Usage


# === Session Management ===

MAX_SESSIONS = 1_000
_sessions: OrderedDict[str, AgentSession] = OrderedDict()
_session_locks: dict[str, Lock] = {}
_sessions_lock = Lock()

def _get_or_create_session(user_id: Optional[str] = None) -> tuple[str, AgentSession, Lock]:
    if user_id and user_id in _sessions:
        _sessions.move_to_end(user_id)
        return user_id, _sessions[user_id], _session_locks[user_id]
    sid = user_id or str(uuid.uuid4())
    session = _agent.create_session()
    _sessions[sid] = session
    session_lock = Lock()
    _session_locks[sid] = session_lock
    if len(_sessions) > MAX_SESSIONS:
        expired_sid, _ = _sessions.popitem(last=False)
        _session_locks.pop(expired_sid, None)
    return sid, session, session_lock


# === OpenAI /v1/chat/completions ===

def _convert_openai_to_agent(messages: list[ChatCompletionMessage]) -> str:
    """Build a user message from the OpenAI-format message list.

    The **last** user message becomes the primary query, and any preceding
    system message is prefixed as context so the agent respects caller
    instructions (e.g. custom output formats).
    """
    system_parts = [msg.content for msg in messages if msg.role == "system"]
    user_parts = [msg.content for msg in messages if msg.role == "user"]

    if not user_parts:
        return ""

    prefix = ""
    if system_parts:
        prefix = "【系统指令】" + "；".join(system_parts) + "\n\n"
    return prefix + user_parts[-1]


@app.post("/v1/chat/completions")
def chat_completions(request: ChatCompletionRequest):
    user_content = _convert_openai_to_agent(request.messages)

    if not user_content:
        raise HTTPException(status_code=400, detail="消息中缺少 user 角色消息")

    try:
        with _sessions_lock:
            sid, session, session_lock = _get_or_create_session(request.user)
        with session_lock:
            # Seed STM with prior conversation when the session is fresh.
            if len(session.stm.messages) <= 1 and len(request.messages) > 1:
                for msg in request.messages[:-1]:
                    if msg.role in ("user", "assistant"):
                        session.stm.add(msg.role, msg.content)

            if request.stream:
                # True SSE streaming — tokens from the final LLM turn are
                # relayed as they are produced (tool-call loops run internally).
                chat_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
                created = int(time.time())

                def _stream_from_agent():
                    yield _sse_chunk(chat_id, created, request.model,
                                     {"role": "assistant", "content": ""}, None)
                    for token in session.process_message_stream(
                        user_content, temperature=request.temperature
                    ):
                        yield _sse_chunk(chat_id, created, request.model,
                                         {"content": token}, None)
                    yield _sse_chunk(chat_id, created, request.model, {}, "stop")
                    yield "data: [DONE]\n\n"

                return StreamingResponse(
                    _stream_from_agent(), media_type="text/event-stream"
                )

            reply = session.process_message(user_content, temperature=request.temperature)
    except Exception:
        logger.exception("Chat completion failed")
        raise HTTPException(status_code=502, detail="上游模型服务暂时不可用，请稍后重试")

    chat_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    return ChatCompletionResponse(
        id=chat_id,
        created=created,
        model=request.model,
        choices=[
            Choice(
                index=0,
                message=ChatCompletionMessage(role="assistant", content=reply),
                finish_reason="stop"
            )
        ],
        usage=Usage(total_tokens=0)
    )


def _sse_chunk(chat_id: str, created: int, model: str,
               delta: dict, finish_reason: Optional[str]) -> str:
    chunk = {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]
    }
    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"


# === Helper endpoints ===

@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "tsinghua-training-plan-advisor",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "tsinghua"
            }
        ]
    }


@app.get("/")
def root():
    return {
        "service": "Tsinghua Training Plan Advisor",
        "api_style": "OpenAI-compatible",
        "endpoints": {
            "chat": "POST /v1/chat/completions",
            "models": "GET /v1/models",
            "programs_list": "GET /programs",
            "programs_detail": "GET /programs/{name}"
        }
    }


@app.get("/programs")
def list_programs():
    items = []
    for m in _programs:
        items.append({
            "name": m.name,
            "department": m.department,
            "total_credits": m.total_credits,
            "degree": m.degree,
            "duration": m.duration,
            "contact": m.contact
        })
    return {"programs": items}


@app.get("/programs/{name}")
def get_program(name: str):
    for m in _programs:
        if name in m.name or m.name in name:
            return {
                "name": m.name,
                "department": m.department,
                "total_credits": m.total_credits,
                "degree": m.degree,
                "duration": m.duration,
                "prerequisites": m.prerequisites,
                "major_restrictions": m.major_restrictions,
                "contact": m.contact
            }
    raise HTTPException(status_code=404, detail=f"未找到培养方案: {name}")
