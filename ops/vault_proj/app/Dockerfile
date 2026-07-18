FROM python:3.12-slim

# Non-root lab image
RUN useradd --create-home --uid 10001 appuser
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py launch.sh .

USER 10001
EXPOSE 8080

# TODO(you): add a real readiness check that fails if Vault is unreachable
# once /secret no longer uses the scaffold fallback — TASKS.md §3
CMD ["/app/launch.sh"]
