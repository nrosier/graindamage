# syntax=docker/dockerfile:1

# --- build stage: resolve and install dependencies into a self-contained venv ---
# Chainguard's python image is Wolfi-based (glibc, apk, minimal package set, rebuilt
# daily) rather than Debian/apt, which is where python:3.14-slim's CVEs came from.
# uv ships in the -dev variant, so there's no need to copy it in from elsewhere.
FROM cgr.dev/chainguard/python:latest-dev AS builder

# The base image defaults to the nonroot user; the builder just needs somewhere writable.
USER root

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
FROM cgr.dev/chainguard/python:latest-dev AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH"

# ffmpeg carries ffprobe, which the CLI runs on the file it is given. It is the only apk
# package this image needs and no smaller Wolfi package carries ffprobe — but it is also
# most of the image's growth, same as it was on Debian. The base image starts as the
# "nonroot" user, so apk needs root, then execution drops back to nonroot below.
USER root
RUN apk add --no-cache ffmpeg

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
USER nonroot

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD ["python", "-c", "import urllib.request as u; u.urlopen('http://127.0.0.1:8080/healthz', timeout=3)"]

# The base image's own ENTRYPOINT runs python directly; clear it so CMD is the full command.
ENTRYPOINT []
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
