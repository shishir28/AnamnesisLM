# CPU serving image. A CUDA serving image can be added after GPU validation.
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TIKTOKEN_CACHE_DIR=/opt/tiktoken-cache
WORKDIR /app
# Keep the large CPU wheel in a layer independent of application edits.
RUN pip install --no-cache-dir "torch>=2.4,<3" --index-url https://download.pytorch.org/whl/cpu
COPY llm/pyproject.toml llm/README.md /app/llm/
COPY llm/src /app/llm/src
COPY api/pyproject.toml /app/api/
COPY api/src /app/api/src
RUN pip install --no-cache-dir /app/llm /app/api \
    && python -c "import tiktoken; tiktoken.get_encoding('gpt2')"
RUN useradd --create-home --uid 10001 app
USER app
ENV ANAMNESIS_DEVICE=cpu ANAMNESIS_CHECKPOINT_PATH=/checkpoints/best.pt
EXPOSE 8000
CMD ["uvicorn", "anamnesis_api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-access-log"]
