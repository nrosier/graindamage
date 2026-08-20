# syntax=docker/dockerfile:1

# --- build stage: resolve and install dependencies into a self-contained venv ---
FROM python:3.13-slim AS builder

# uv comes from its official image, so no local uv install is needed.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    VIRTUAL_ENV=/opt/venv

WORKDIR /src
RUN uv venv "$VIRTUAL_ENV"

# Dependencies first: this layer is cached until pyproject.toml changes.
COPY pyproject.toml README.md ./
RUN uv pip install --no-cache -r pyproject.toml

# Then the app itself (templates and static assets travel with the package).
COPY app ./app
RUN uv pip install --no-cache --no-deps .

# --- runtime stage ---
FROM python:3.13-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH"

RUN groupadd --system app && useradd --system --gid app --no-create-home app

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
USER app

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8080/healthz', timeout=3)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
