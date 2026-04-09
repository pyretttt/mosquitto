# OpenBB Telegram cron worker.
# Required env: BOT_TOKEN, CHAT_ID, PERIOD_SEC
FROM python:3.13-slim

# This flag is important to output python logs correctly in docker!
# This flag is important to output python logs correctly in docker!
ENV PYTHONUNBUFFERED=1
# Flag to optimize container size a bit by removing runtime python cache

ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

ARG REQUIREMENTS_FILE=requirements.txt
COPY requirements.txt requirements.txt
COPY requirements-dev.txt requirements-dev.txt

RUN python -m pip install -r ${REQUIREMENTS_FILE}

COPY ./configs/ ./configs/
COPY ./src/ ./src/


ENV BOT_TOKEN="" \
    CHAT_ID="" \
    DRY_RUN=false \
    RUNLOOP_INTERVAL_SEC=60


CMD ["python", "-m", "src.app"]
