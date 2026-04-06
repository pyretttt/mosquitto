# OpenBB Telegram cron worker.
# Required env: BOT_TOKEN, CHAT_ID, PERIOD_SEC
FROM python:3.13-slim-buster

# This flag is important to output python logs correctly in docker!
ENV PYTHONUNBUFFERED 1
# Flag to optimize container size a bit by removing runtime python cache
ENV PYTHONDONTWRITEBYTECODE 1

WORKDIR /app

COPY requirements.txt ./

RUN python -m venv .venv && source .venv/bin/activate && pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY ./src/ ./src/

CMD ["python", "app.py"]
