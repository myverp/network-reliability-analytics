FROM python:3.13-slim

WORKDIR /app

COPY software/pyproject.toml software/README.md ./
COPY software/src ./src
RUN pip install --no-cache-dir .

COPY alembic.ini ./
COPY software ./software
EXPOSE 8000

CMD ["sh", "-c", "alembic upgrade head && uvicorn functional_stability.api.main:app --host 0.0.0.0 --port 8000"]
