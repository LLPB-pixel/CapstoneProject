# Dockerfile for Prompt Guard - AI Security System
# Simple, production-ready image

FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV APP_HOME=/app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ca-certificates \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create and set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the application
COPY . /app

# Create a non-root user
RUN useradd -m -u 1000 -g 1000 appuser

# Create data directory for SQLite database
RUN mkdir -p /app/data && chown -R appuser:appuser /app/data

USER appuser

# Expose ports
EXPOSE 8000  # FastAPI backend

# Volume for persistent attack database
VOLUME /app/data

# Health check
HEALTHCHECK --interval=30s --timeout=3s \
    CMD curl -f http://localhost:8000/health || exit 1

# Command to run the application
CMD ["sh", "-c", "cd /app/3-LLM-judge && python api_server.py ${MISTRAL_API_KEY:-demo} --port 8000"]
