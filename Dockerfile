FROM python:3.12-slim

WORKDIR /app

# 安装依赖（利用 Docker 层缓存）
COPY requirements.txt .
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt


# 创建非 root 用户
RUN addgroup --system --gid 1001 appuser \
    && adduser --system --uid 1001 --gid 1001 appuser

# 复制源码
COPY agent/ agent/
COPY api/ api/
COPY run.py .
COPY 2025级培养方案.md .
COPY curated_courses.json .

# 创建数据缓存目录，设为全局可写以便兼容宿主机 volume 挂载
RUN mkdir -p /app/data && chmod 777 /app/data

# 预下载词嵌入模型（构建时下载到共享缓存，避免运行时等待）
ENV HF_ENDPOINT=https://hf-mirror.com
ENV SENTENCE_TRANSFORMERS_HOME=/app/model_cache
RUN mkdir -p /app/model_cache && chmod 777 /app/model_cache \
    && python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('shibing624/text2vec-base-chinese')"

USER appuser

EXPOSE 8000

ENV PYTHONUNBUFFERED=1
ENV LANG=C.UTF-8
ENV HF_ENDPOINT=https://hf-mirror.com
ENV SENTENCE_TRANSFORMERS_HOME=/app/model_cache

CMD ["python", "run.py", "--mode", "api", "--host", "0.0.0.0", "--port", "8000"]
