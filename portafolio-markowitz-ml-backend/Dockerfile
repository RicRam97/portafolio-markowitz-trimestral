FROM python:3.11-slim

# The env var PORT is provided by Railway at runtime
ENV PORT=8000
ENV HOST=0.0.0.0

WORKDIR /app

# Ensure gcc and other build tools are available for scipy / cvxpy
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# Run uvicorn on the port specified by the PORT environment variable
CMD ["sh", "-c", "uvicorn app.main:app --host $HOST --port $PORT"]
