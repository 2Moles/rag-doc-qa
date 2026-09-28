FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[claude]"
COPY data ./data

RUN useradd --create-home app
USER app

EXPOSE 8000
# Pass ANTHROPIC_API_KEY at run time (docker run -e ANTHROPIC_API_KEY=...) to enable Claude answers.
CMD ["uvicorn", "ragqa.api:app", "--host", "0.0.0.0", "--port", "8000"]
