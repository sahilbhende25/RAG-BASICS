# Multi-stage-ish slim image for the LexRAG API.
FROM python:3.13-slim

# Avoid .pyc, unbuffered logs (so container logs stream immediately).
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install deps first so this layer caches unless requirements change.
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Then copy the app (changes here don't bust the deps layer).
COPY lexrag ./lexrag
COPY data ./data

EXPOSE 8000

# OPENAI_API_KEY must be provided at runtime, e.g.:
#   docker run -e OPENAI_API_KEY=sk-... -p 8000:8000 lexrag
CMD ["uvicorn", "lexrag.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
