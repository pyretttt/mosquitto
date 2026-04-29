# Small image just for the MLflow server. Using the ghcr image would also work
# but building our own makes it obvious what's installed and keeps pins visible.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# sqlalchemy is needed for --backend-store-uri sqlite:///...
RUN pip install --no-cache-dir \
      "mlflow[auth]>=3.11.1,<4.0" \
      "sqlalchemy>=2.0,<3.0" \
      "psycopg2-binary>=2.9,<3.0" \
      "boto3>=1.34,<2.0"


# RUN mkdir -p /mlflow/db /mlflow/artifacts

EXPOSE 5000
