FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    gcc g++ libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first (saves ~2 GB vs default CUDA build)
RUN pip install --no-cache-dir \
    torch==2.2.2 \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create non-root user
RUN useradd -m -u 1000 trader && \
    mkdir -p /app/data/models /app/logs && \
    chown -R trader:trader /app

USER trader

# Copy source
COPY --chown=trader:trader src/ ./src/
COPY --chown=trader:trader main.py .

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -sf http://localhost:8080/health || exit 1

CMD ["python", "main.py", "--mode", "all"]
