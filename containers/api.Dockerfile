# The prediction API (notebook 10b): FastAPI in a container, the same image on this laptop and in Lambda.
#
# In Lambda, the AWS Lambda Web Adapter (an extension, copied in below) turns each invocation into an
# ordinary HTTP request to uvicorn on port 8080, so the app has no Lambda-specific code. On a laptop the
# adapter stays idle and uvicorn answers directly.
#
# Two stages: uv installs the dependencies in the first; the image that ships starts again from the plain
# Python image and takes only the finished environment and our code, so no build tools go to production.
# Both are Debian 13: on Debian 12 the registry's scan found 10 critical and 47 high vulnerabilities, all in
# old system packages (notebook 10b).
#
# Build from the repo root, for Lambda's Graviton (arm64) processors:
#   docker build --platform linux/arm64 -f containers/api.Dockerfile -t qafi-api .

FROM ghcr.io/astral-sh/uv:python3.12-trixie-slim AS build
WORKDIR /app
# Only the API's dependencies (the group "api" in pyproject.toml), from the lock file: no pandas, no
# scikit-learn, so the image is small and starts fast. Compiled now, because Lambda's disk is read-only.
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --only-group api --no-install-project

FROM python:3.12-slim-trixie
COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:1.1.0 /lambda-adapter /opt/extensions/lambda-adapter
WORKDIR /app
COPY --from=build /app/.venv /app/.venv
COPY src/qafi_platform/__init__.py src/qafi_platform/api.py src/qafi_platform/athena.py ./qafi_platform/
RUN python -m compileall -q qafi_platform && useradd --create-home app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH=/app \
    AWS_LWA_PORT=8080 \
    AWS_LWA_READINESS_CHECK_PATH=/health
USER app
EXPOSE 8080
CMD ["uvicorn", "qafi_platform.api:app", "--host", "0.0.0.0", "--port", "8080"]
