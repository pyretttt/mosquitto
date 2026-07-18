#!/bin/sh
set -eu
PEM=/vault/secrets/tls.pem
CERT=/tmp/tls.crt
KEY=/tmp/tls.key

awk '/BEGIN CERTIFICATE/,/END CERTIFICATE/' "$PEM" > "$CERT"
awk '/BEGIN .*PRIVATE KEY/,/END .*PRIVATE KEY/' "$PEM" > "$KEY"

exec uvicorn main:app --host 0.0.0.0 --port 8080 \
  --ssl-certfile "$CERT" \
  --ssl-keyfile "$KEY"