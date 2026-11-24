FROM python:3.12.10-slim

COPY --from=ghcr.io/astral-sh/uv:0.13.9 /uv /uvx /bin/

ENV UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

# Non-root user: if anything in the app is ever compromised, the attacker is
# an unprivileged user inside the container, not root.
# /code = app + its .venv (Chainlit writes .files/ there at runtime);
# /state = doc-ingest's "done" marker volume (a new named volume copies this
# directory's ownership, so the app user can write it).
RUN useradd --create-home --uid 10001 app \
 && mkdir -p /code /state \
 && chown app:app /code /state
USER app
WORKDIR /code

# Dependencies first (cached layer, rebuilt only when the lock file changes).
# COPY --chown instead of a later `chown -R`, which would duplicate the 1 GB .venv layer.
COPY --chown=app:app pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev --no-install-project

COPY --chown=app:app . .
RUN uv sync --frozen --no-dev

ENV PATH="/code/.venv/bin:$PATH"

EXPOSE 8000

CMD ["chainlit", "run", "src/supplyguard_ai/ui/chainlit_app.py", "-h", "--host", "0.0.0.0", "--port", "8000"]
