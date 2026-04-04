# OpenBB Telegram cron worker.
# Required env: BOT_TOKEN, CHAT_ID, PERIOD_SEC
# Optional: SYMBOL, PROVIDER, PRICE_ABOVE, PRICE_BELOW, PCT_CHANGE_ABS,
#           ALERT_COOLDOWN_SEC, DRY_RUN
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt ./

RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

ENV PATH="/opt/venv/bin:$PATH"
ENV VIRTUAL_ENV="/opt/venv"

COPY . .

CMD ["python", "app.py"]
