# syntax=docker/dockerfile:1

FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System dependencies (optional; uncomment if needed)
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential \
#  && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first for better layer caching
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy application code
COPY . .

# Cloud Run sets PORT dynamically; default to 8000 for local runs
ENV PORT=8000

EXPOSE 8000

# Start FastAPI app
# Use shell form so $PORT expands at runtime (Render supplies PORT)
CMD uvicorn api:app --host 0.0.0.0 --port $PORT
