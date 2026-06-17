FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md /app/
COPY apps /app/apps

RUN pip install --upgrade pip \
    && pip install -e .[consumer]

ENV PYTHONPATH=/app

CMD ["python", "-u", "apps/consumer/consumer.py"]
