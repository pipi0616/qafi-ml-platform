# Platform image: the engineering code plus its locked dependencies.
# Every job (ingestion, scoring, API) runs from this image, so "works on my laptop"
# and "works in AWS" are the same environment.
#
# Build from the repo root:  docker build -f containers/Dockerfile -t qafi-platform .

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

# Install dependencies first, from the lock file only, so this layer is cached
# until pyproject.toml or uv.lock change.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

# Then the code, which changes often.
COPY src ./src
RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"
RUN useradd --create-home app && chown -R app /app
USER app

CMD ["python", "-c", "import qafi_platform; print('qafi-platform image OK')"]
