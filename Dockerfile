FROM docker.m.daocloud.io/library/python:3.10-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade "pip>=21.0,<26.0" "setuptools>=65.0,<70.0" wheel

RUN pip install --no-cache-dir -r requirements.txt

RUN python3 -c "from silero_vad import load_silero_vad; load_silero_vad(onnx=True)" 2>/dev/null || \
    python3 -c "from silero_vad import load_silero_vad; load_silero_vad()" 2>/dev/null || true


FROM docker.m.daocloud.io/library/python:3.10-slim AS runner

WORKDIR /app

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.10/site-packages/ /usr/local/lib/python3.10/site-packages/
COPY --from=builder /usr/local/bin/ /usr/local/bin/
COPY --from=builder /root/.cache/ /root/.cache/

COPY backend/ /app/backend/
COPY frontend/ /app/frontend/

ENV PYTHONUNBUFFERED=1
ENV UPLOAD_DIR=/app/uploads
ENV OUTPUT_DIR=/app/output
ENV MODEL_DIR=/app/models

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "600", "--workers", "2"]