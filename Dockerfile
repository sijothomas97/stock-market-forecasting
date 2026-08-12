# Time series forecasting API + dashboard.
#
# Build:  docker build -t ts-stocks-api .
# Run:    docker run --rm -p 8000:8000 -v "$(pwd)/data:/app/data:ro" ts-stocks-api
#
# Data is not baked into the image (630MB of NSE CSVs, gitignored — see
# data/README.md); mount a populated data/ directory at runtime. Without a
# mount, /stocks and /health still respond (0 symbols available), useful
# for verifying the container boots in CI without real data.
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps for scientific Python wheels (statsmodels/arch build against
# these on some platforms; slim base doesn't ship a compiler by default).
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config.yaml .
COPY src/ src/
COPY static/ static/

# Placeholder so the app boots without a mounted data/ volume.
RUN mkdir -p data/stocks_data

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
