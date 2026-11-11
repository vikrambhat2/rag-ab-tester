# syntax=docker/dockerfile:1
FROM python:3.12-slim AS base

COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Dependencies first for better layer caching (CLI only: no UI extra, no dev group).
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
COPY experiments ./experiments
COPY data/docs ./data/docs
COPY ingest.py run_experiment.py run_all.py ./

# Run as a non-root user; results/ and data/ are the only writable paths.
RUN useradd --create-home --uid 1000 app \
    && mkdir -p results data \
    && chown -R app:app /app/results /app/data
USER app

# Credentials are provided at runtime (--env-file .env), never baked into the image.
ENTRYPOINT []
CMD ["python", "run_all.py", "--save-json"]
