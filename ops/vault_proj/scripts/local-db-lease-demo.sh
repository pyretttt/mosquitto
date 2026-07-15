#!/usr/bin/env bash
# Optional stretch: Postgres in Docker + Vault database secrets engine.
# Issues dynamic credentials, then renew / revoke the lease.
#
# Requires: vault server -dev running, VAULT_ADDR + VAULT_TOKEN set, Docker.
# Optional: psql client (script falls back to docker exec).
set -euo pipefail

: "${VAULT_ADDR:?Set VAULT_ADDR (e.g. http://127.0.0.1:8200)}"
: "${VAULT_TOKEN:?Set VAULT_TOKEN to the dev root token}"

PG_CONTAINER="${PG_CONTAINER:-vault-lab-pg}"
PG_IMAGE="${PG_IMAGE:-postgres:16-alpine}"
PG_PORT="${PG_PORT:-5432}"
PG_USER="${PG_USER:-vault}"
PG_PASSWORD="${PG_PASSWORD:-vault-root-lab-only}"
PG_DB="${PG_DB:-vaultlab}"
ROLE_NAME="${ROLE_NAME:-app}"
# Short TTLs so renew/revoke are obvious in a lab session.
DEFAULT_TTL="${DEFAULT_TTL:-3m}"
MAX_TTL="${MAX_TTL:-10m}"

psql_as() {
  local user="$1" password="$2"
  shift 2
  if command -v psql >/dev/null 2>&1; then
    PGPASSWORD="$password" psql -h 127.0.0.1 -p "$PG_PORT" -U "$user" -d "$PG_DB" "$@"
  else
    docker exec -e PGPASSWORD="$password" -i "$PG_CONTAINER" \
      psql -h 127.0.0.1 -U "$user" -d "$PG_DB" "$@"
  fi
}

echo "==> Vault status"
vault status

echo "==> Start Postgres ($PG_CONTAINER) on localhost:$PG_PORT"
if docker ps -a --format '{{.Names}}' | grep -qx "$PG_CONTAINER"; then
  docker start "$PG_CONTAINER" >/dev/null
else
  docker run -d --name "$PG_CONTAINER" \
    -e POSTGRES_USER="$PG_USER" \
    -e POSTGRES_PASSWORD="$PG_PASSWORD" \
    -e POSTGRES_DB="$PG_DB" \
    -p "${PG_PORT}:5432" \
    "$PG_IMAGE" >/dev/null
fi

echo "==> Wait for Postgres ready"
for _ in $(seq 1 30); do
  if docker exec "$PG_CONTAINER" pg_isready -U "$PG_USER" -d "$PG_DB" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
docker exec "$PG_CONTAINER" pg_isready -U "$PG_USER" -d "$PG_DB"

# Rest can be done by hands:

# echo "==> Enable database secrets engine (ok if already enabled)"
# vault secrets enable database 2>/dev/null || true

# echo "==> Configure Postgres connection (Vault uses these root creds to CREATE/DROP users)"
# # host.docker.internal is NOT needed: Vault runs on the host and talks to published :5432.
# vault write database/config/postgres \
#   plugin_name=postgresql-database-plugin \
#   allowed_roles="$ROLE_NAME" \
#   connection_url="postgresql://{{username}}:{{password}}@127.0.0.1:${PG_PORT}/${PG_DB}?sslmode=disable" \
#   username="$PG_USER" \
#   password="$PG_PASSWORD"

# echo "==> Create dynamic role '$ROLE_NAME' (ttl=$DEFAULT_TTL max_ttl=$MAX_TTL)"
# vault write "database/roles/$ROLE_NAME" \
#   db_name=postgres \
#   creation_statements="CREATE ROLE \"{{name}}\" WITH LOGIN PASSWORD '{{password}}' VALID UNTIL '{{expiration}}'; \
# GRANT SELECT ON ALL TABLES IN SCHEMA public TO \"{{name}}\";" \
#   default_ttl="$DEFAULT_TTL" \
#   max_ttl="$MAX_TTL"

# echo "==> Issue dynamic credentials"
CREDS_JSON=$(vault read -format=json "database/creds/$ROLE_NAME")
LEASE_ID=$(echo "$CREDS_JSON" | python3 -c 'import json,sys; print(json.load(sys.stdin)["lease_id"])')
DYN_USER=$(echo "$CREDS_JSON" | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"]["username"])')
DYN_PASS=$(echo "$CREDS_JSON" | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"]["password"])')
LEASE_DURATION=$(echo "$CREDS_JSON" | python3 -c 'import json,sys; print(json.load(sys.stdin)["lease_duration"])')

# echo "    username: $DYN_USER"
# echo "    password: $DYN_PASS"
# echo "    lease_id: $LEASE_ID"
# echo "    lease_duration: ${LEASE_DURATION}s"

# echo "==> Verify login works with dynamic user"
# psql_as "$DYN_USER" "$DYN_PASS" -c 'SELECT current_user, now();'

# echo "==> Renew lease (+30s)"
# vault lease renew -increment=30s "$LEASE_ID"

# echo "==> Revoke lease (Postgres role should be dropped)"
# vault lease revoke "$LEASE_ID"

# echo "==> Verify login FAILS after revoke (expected)"
# if psql_as "$DYN_USER" "$DYN_PASS" -c 'SELECT 1;' >/dev/null 2>&1; then
#   echo "ERROR: credential still works after revoke"
#   exit 1
# else
#   echo "Good: dynamic user can no longer connect."
# fi

# echo ""
# echo "Done. Manual follow-ups:"
# echo "  vault read database/creds/$ROLE_NAME          # issue another pair"
# echo "  vault lease renew <lease_id>"
# echo "  vault lease revoke <lease_id>"
# echo "  docker stop $PG_CONTAINER && docker rm $PG_CONTAINER   # cleanup"
