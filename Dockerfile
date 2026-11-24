FROM python:3.12.10-slim

COPY --from=ghcr.io/astral-sh/uv:0.13.9 /uv /uvx /bin/

ENV UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1

WORKDIR /code

COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev --no-install-project

COPY . .
RUN uv sync --frozen --no-dev

ENV PATH="/code/.venv/bin:$PATH"

EXPOSE 8000

CMD ["chainlit", "run", "src/supplyguard_ai/ui/chainlit_app.py", "-h", "--host", "0.0.0.0", "--port", "8000"]
