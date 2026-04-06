# OpenBB Telegram cron worker.
# Required env: BOT_TOKEN, CHAT_ID, PERIOD_SEC
FROM python:3.13-slim-buster

# This flag is important to output python logs correctly in docker!
# This flag is important to output python logs correctly in docker!
ENV PYTHONUNBUFFERED=1
# Flag to optimize container size a bit by removing runtime python cache

ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

ARG REQUIREMENTS_FILE=requirements.txt
COPY ${REQUIREMENTS_FILE} ./requirements.txt

RUN python -m venv .venv \
    && .venv/bin/pip install --no-cache-dir --upgrade pip \
    && .venv/bin/pip install --no-cache-dir -r requirements.txt

COPY ./src/ ./src/
COPY cfg.yaml ./

ENV BOT_TOKEN="" \
    CHAT_ID=""


CMD [".venv/bin/python", "-m", "src.app"]
