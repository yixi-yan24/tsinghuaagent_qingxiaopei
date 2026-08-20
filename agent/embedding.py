import os, json, pickle, threading
import numpy as np
from typing import Optional
from collections import OrderedDict

os.environ["HF_ENDPOINT"] = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")

MODEL_NAME = "shibing624/text2vec-base-chinese"
EMBEDDING_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
EMBEDDING_CACHE_FILE = os.path.join(EMBEDDING_CACHE_DIR, "embeddings.pkl")
EMBEDDING_META_FILE = os.path.join(EMBEDDING_CACHE_DIR, "embeddings_meta.json")

# 模型 + 索引都是进程级单例，加锁保护并发首次初始化。
_model = None
_model_lock = threading.Lock()
_index_cache = None            # 驻留内存的 (embeddings, texts, program_ids)
_index_lock = threading.Lock()

# 语义搜索并发限制：encode 是 CPU 密集 + GIL 阻塞，2 核下同时 2 个已是上限。
_ENCODE_SEM = threading.Semaphore(2)

# 查询编码结果小缓存（LRU）——相同/相近 query 直接复用，避免重复推理。
_QUERY_CACHE: "OrderedDict[str, np.ndarray]" = OrderedDict()
_QUERY_CACHE_MAX = 128
_QUERY_CACHE_LOCK = threading.Lock()


def _load_model():
    """Load the sentence-transformer model exactly once (thread-safe).

    Suppresses the tqdm "Loading weights: …" progress bar so it never pollutes
    CLI output or API logs.  Use ``TQDM_DISABLE=0`` to force it back on.
    """
    global _model
    if _model is not None:
        return _model
    with _model_lock:
        if _model is not None:
            return _model
        os.environ.setdefault("TQDM_DISABLE", "1")
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME)
        return _model


def warmup(programs: Optional[list] = None) -> None:
    """Preload the model and the vector index at startup.

    Call this once in a background thread so the first semantic_search does
    not pay the one-time model download / load latency.
    """
    _load_model()
    if programs:
        get_or_build_index(programs)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def compute_embeddings(texts: list[str]) -> np.ndarray:
    """Compute embeddings for a list of texts."""
    model = _load_model()
    return model.encode(texts, show_progress_bar=True, normalize_embeddings=True)


def _encode_query(query: str) -> np.ndarray:
    """Encode one query with a small LRU cache + a global concurrency cap.

    ``model.encode`` is the expensive part (runs the transformer on CPU); the
    cache avoids recomputing for repeated questions, and the semaphore keeps
    concurrent semantic searches from saturating a 2-core box.
    """
    with _QUERY_CACHE_LOCK:
        if query in _QUERY_CACHE:
            _QUERY_CACHE.move_to_end(query)
            return _QUERY_CACHE[query]

    model = _load_model()
    with _ENCODE_SEM:
        emb = model.encode([query], normalize_embeddings=True)[0]

    with _QUERY_CACHE_LOCK:
        _QUERY_CACHE[query] = emb
        if len(_QUERY_CACHE) > _QUERY_CACHE_MAX:
            _QUERY_CACHE.popitem(last=False)
    return emb


def build_index(programs: list) -> tuple[np.ndarray, list[str]]:
    """Build embedding index from training programs.

    Returns:
        embeddings: (N, D) numpy array, normalized
        texts: list of N search texts
    """
    texts = []
    for m in programs:
        combined = (
            f"{m.department} {m.name} "
            f"{m.major_restrictions[:200]} "
            f"{m.prerequisites[:200]} "
        )
        texts.append(combined)
    embeddings = compute_embeddings(texts)
    return embeddings, texts


def save_index(embeddings: np.ndarray, texts: list[str], program_ids: list[str]):
    """Save embedding index to disk."""
    os.makedirs(EMBEDDING_CACHE_DIR, exist_ok=True)
    with open(EMBEDDING_CACHE_FILE, "wb") as f:
        pickle.dump({"embeddings": embeddings, "texts": texts, "program_ids": program_ids}, f)
    with open(EMBEDDING_META_FILE, "w", encoding="utf-8") as f:
        json.dump({"texts": texts, "program_ids": program_ids}, f, ensure_ascii=False, indent=2)


def load_index() -> Optional[tuple[np.ndarray, list[str], list[str]]]:
    """Load saved embedding index from disk."""
    if not os.path.exists(EMBEDDING_CACHE_FILE):
        return None
    try:
        with open(EMBEDDING_CACHE_FILE, "rb") as f:
            data = pickle.load(f)
        return data["embeddings"], data["texts"], data["program_ids"]
    except Exception:
        return None


def get_or_build_index(programs: list, force_rebuild: bool = False
                       ) -> tuple[np.ndarray, list[str], list[str]]:
    """Get the index, preferring the in-memory copy, then disk, then build.

    The built index stays resident in ``_index_cache`` so repeated queries
    never re-read from disk or re-run the model over all programs.
    """
    global _index_cache
    if not force_rebuild and _index_cache is not None and len(_index_cache[2]) == len(programs):
        return _index_cache

    if not force_rebuild:
        cached = load_index()
        if cached is not None and len(cached[2]) == len(programs):
            _index_cache = cached
            return cached

    with _index_lock:
        # Double-checked under lock in case another thread built it meanwhile.
        if _index_cache is not None and len(_index_cache[2]) == len(programs):
            return _index_cache
        program_ids = [m.name for m in programs]
        embeddings, texts = build_index(programs)
        try:
            save_index(embeddings, texts, program_ids)
        except Exception:
            pass
        _index_cache = (embeddings, texts, program_ids)
        return _index_cache


def semantic_search(query: str, programs: list, top_k: int = 5) -> list[tuple]:
    """Search programs by semantic similarity to query.

    Returns list of (program, score) tuples, sorted by relevance descending.
    The heavy part (model.encode) is concurrency-capped and cached; the
    dot-product scoring is vectorised across the whole index in one call.
    """
    embeddings, texts, program_ids = get_or_build_index(programs)
    query_emb = _encode_query(query)

    # 矩阵乘一次算完与所有培养方案的点积（已归一化 → 即余弦相似度）。
    scores = embeddings @ query_emb
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [(programs[i], float(scores[i])) for i in top_idx]
