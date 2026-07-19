#!/usr/bin/env bash
# §4 — set short TTLs on the demo secret / auth role for renew+revoke practice.
# Requires: reachable Vault, VAULT_ADDR + VAULT_TOKEN (root or admin).
#
#   kubectl -n vault port-forward svc/vault 8200:8200
#   export VAULT_ADDR=http://127.0.0.1:8200 VAULT_TOKEN=<init-root-token>
#   mise run ttl-lab
set -euo pipefail

: "${VAULT_ADDR:?Set VAULT_ADDR to reachable Vault API}"
: "${VAULT_TOKEN:?Set VAULT_TOKEN}"
NS_APP="${NS_APP:-demo}"
SA_NAME="${SA_NAME:-demo-app}"
# Short enough to demo expiry; Agent renews before this elapses.
ROLE_TTL="${ROLE_TTL:-2m}"
ROLE_MAX_TTL="${ROLE_MAX_TTL:-10m}"
# KV version auto-delete (lab only). Keep longer than ROLE_TTL so the app
# stays readable while you practice token renew/revoke.
KV_DELETE_AFTER="${KV_DELETE_AFTER:-24h}"

echo "==> Vault status"
vault status

echo "==> KV v2 custom metadata + delete-version-after on secret/demo/db"
# custom_metadata is labels (not enforced by itself).
# -delete-version-after is real auto-expiry of KV versions.
vault kv metadata put \
  -custom-metadata=owner=demo-app \
  -custom-metadata=rotation_hint="${ROLE_TTL}" \
  -delete-version-after="$KV_DELETE_AFTER" \
  secret/demo/db

echo "==> Metadata:"
vault kv metadata get secret/demo/db

echo "==> Short TTL on kubernetes auth role demo-app (ttl=${ROLE_TTL} max_ttl=${ROLE_MAX_TTL})"
vault write auth/kubernetes/role/demo-app \
  bound_service_account_names="$SA_NAME" \
  bound_service_account_namespaces="$NS_APP" \
  policies=demo-app,pki-demo-app \
  ttl="$ROLE_TTL" \
  max_ttl="$ROLE_MAX_TTL"

echo "==> Role:"
vault read auth/kubernetes/role/demo-app

echo "==> Short renewable CLI token (for renew/revoke before touching the app)"
CHILD=$(vault token create -policy=demo-app -ttl="$ROLE_TTL" -renewable=true -format=json)
CHILD_TOKEN=$(echo "$CHILD" | python3 -c 'import json,sys; print(json.load(sys.stdin)["auth"]["client_token"])')
CHILD_ACCESSOR=$(echo "$CHILD" | python3 -c 'import json,sys; print(json.load(sys.stdin)["auth"]["accessor"])')

echo "    client_token (keep private): $CHILD_TOKEN"
echo "    accessor: $CHILD_ACCESSOR"
echo ""
echo "Next (TASKS.md §4 renew/revoke):"
echo "  VAULT_TOKEN=$CHILD_TOKEN vault token lookup"
echo "  VAULT_TOKEN=$CHILD_TOKEN vault token renew -increment=${ROLE_TTL}"
echo "  vault token revoke -accessor $CHILD_ACCESSOR"
echo "  # then: VAULT_TOKEN=$CHILD_TOKEN vault kv get secret/demo/db  → should fail"
echo ""
echo "App tokens use the kubernetes role TTL above; restart demo-app to pick up a new lease:"
echo "  kubectl -n $NS_APP rollout restart deploy/demo-app"
