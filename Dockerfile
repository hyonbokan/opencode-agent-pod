# syntax=docker/dockerfile:1

# opencode-agent-pod: the FastAPI/SSE service bundled with a pinned opencode binary.
FROM python:3.12-slim

# Pinned opencode version. Bump deliberately, then re-verify a real run before shipping.
ARG OPENCODE_VERSION=1.17.11

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/home/pod \
    AGENT_POD_HOST=0.0.0.0 \
    AGENT_POD_PORT=8080

# Node.js is needed both to install the opencode CLI package and at runtime, where opencode installs
# provider SDK packages for custom providers. curl stays in the image for the healthcheck.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates gnupg \
    && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && npm install -g opencode-ai@${OPENCODE_VERSION} \
    && npm cache clean --force \
    && apt-get purge -y gnupg \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first so the layer caches across app-code changes.
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Application code. Tests, scripts, docs, and secrets are excluded via .dockerignore.
COPY agent/ ./agent/
COPY core/ ./core/
COPY llm/ ./llm/
COPY config/ ./config/
COPY pod/ ./pod/

# The pod runs LLM-authored shell commands, so it runs as a non-root user with a writable home for
# opencode's own data and cache directories.
RUN useradd --create-home --uid 10001 pod \
    && chown -R pod:pod /app /home/pod
USER pod

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
    CMD curl -fsS http://localhost:8080/health || exit 1

# One autonomous-agent service. Provide AGENT_POD_TOKEN and provider keys at runtime as secrets.
CMD ["python", "-m", "pod"]
