FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1

WORKDIR /app

# System deps for OpenCV and PaddleOCR
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git \
    libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install uv (Astral) package manager
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && ln -s /root/.local/bin/uv /usr/local/bin/uv


# Copy project metadata first (better layer caching)
COPY pyproject.toml ./
COPY README.md ./

# Install dependencies using uv
RUN uv pip install .

# Copy source
COPY app ./app
COPY config ./config

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]


