# SecPipeAI — CPU-only reproducibility image
# Python 3.12-slim (matches production environment: Python 3.12.3, Linux x86_64)
#
# USAGE
# -----
# Build:
#   docker build -t secpipeai .
#
# Run full pipeline (datasets must be mounted into /app/data/raw/):
#   docker run --rm \
#     -v /path/to/cicids2017/csvs:/app/data/raw/cicids2017 \
#     -v /path/to/unsw_nb15/csvs:/app/data/raw/unsw_nb15 \
#     -v $(pwd)/outputs:/app/outputs \
#     secpipeai make all
#
# Quick smoke-test (no data required — runs make help):
#   docker run --rm secpipeai
#
# DATASET NOTE
# ------------
# CICIDS2017 and UNSW-NB15 cannot be downloaded automatically.
# Mount them as read-only volumes (see USAGE above).
# See src/data/download.py for exact filenames and SHA-256 checksums.

FROM python:3.12-slim-bookworm

# Metadata
LABEL maintainer="nnolas27@asu.edu"
LABEL description="SecPipeAI: Reproducible anomaly detection baseline evaluation"
LABEL version="1.0"

# Reproducibility: fix Python hash seed and disable .pyc files
ENV PYTHONHASHSEED=0
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# System deps: make (for Makefile targets), gcc (for some pip builds)
RUN apt-get update && apt-get install -y --no-install-recommends \
        make \
        gcc \
        g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer-cached unless requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy project source
COPY configs/   configs/
COPY src/       src/
COPY Makefile   .

# Create expected directory structure with .gitkeep placeholders
RUN mkdir -p data/raw/cicids2017 \
             data/raw/unsw_nb15 \
             data/processed \
             outputs/models \
             outputs/metrics \
             outputs/figures \
             outputs/tables \
             outputs/paper \
             docs

# Run as non-root user for security
RUN useradd --create-home --shell /bin/bash secpipeai \
 && chown -R secpipeai:secpipeai /app
USER secpipeai

# Default: show available targets (safe no-op when no data is mounted)
CMD ["make", "help"]
