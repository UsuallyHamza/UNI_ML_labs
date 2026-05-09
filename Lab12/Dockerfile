# We explicitly target amd64 to ensure compatibility with AWS as my system is ARM64 M1
FROM --platform=linux/amd64 python:3.12-slim

WORKDIR /app

# Install uv inside the container for fast dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy project files
COPY pyproject.toml uv.lock ./
COPY app.py model.joblib ./
COPY templates/ ./templates/

# Let uv create the optimized .venv inside the container
RUN uv sync --no-dev

EXPOSE 5000

# Use 'uv run' to automatically execute within the isolated environment
CMD ["uv", "run", "python", "app.py"]