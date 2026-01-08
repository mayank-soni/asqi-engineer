# Stage 1: Builder
FROM ghcr.io/astral-sh/uv:bookworm-slim AS builder

WORKDIR /app

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Configure uv for faster builds
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
ENV UV_PYTHON_INSTALL_DIR=/python
ENV UV_PYTHON_PREFERENCE=only-managed

# Install Python
RUN uv python install 3.13

# Copy only dependency files first (for better layer caching)
COPY pyproject.toml uv.lock* ./

# Install dependencies only (no project installation yet)
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,target=/root/.cache/pip \
    uv sync --frozen --no-install-project

# Now copy the rest of the project
COPY . .

# Install the project itself
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen

# Download NLTK data required for tokenization in parallel with other operations
RUN --mount=type=cache,target=/root/.cache/pip \
    uv run python -c "import nltk; nltk.download('punkt_tab', quiet=True)" &

# Wait for background NLTK download if needed
RUN wait || true

# Stage 2: Runtime
FROM debian:bookworm-slim

WORKDIR /app

# Copy venv and Python installation from builder
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /python /python
ENV PATH="/app/.venv/bin:$PATH"

COPY manifest.yaml .
COPY entrypoint.py .
COPY *.py ./

RUN chmod +x entrypoint.py

ENTRYPOINT ["python", "./entrypoint.py"]