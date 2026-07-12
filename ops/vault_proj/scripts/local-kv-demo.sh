#!/usr/bin/env bash
# Guided local demo: KV v2 + limited policy token.
# Requires: vault server -dev running, VAULT_ADDR + VAULT_TOKEN set.
set -euo pipefail

: "${VAULT_ADDR:?Set VAULT_ADDR (e.g. http://127.0.0.1:8200)}"
: "${VAULT_TOKEN:?Set VAULT_TOKEN to the dev root token}"

echo "==> Vault status"
vault status

echo "==> Enable KV v2 at path secret/ (ok if already enabled)"
vault secrets enable -path=secret kv-v2 2>/dev/null || true

echo "==> Write demo secret"
vault kv put secret/demo/db username=app password="local-dev-only-change-me"

echo "==> Read demo secret"
vault kv get secret/demo/db

POLICY_FILE="$(mktemp)"
trap 'rm -f "$POLICY_FILE"' EXIT

# Minimal read-only policy for KV v2 (note the /data/ prefix).
cat >"$POLICY_FILE" <<'EOF'
path "secret/data/demo/*" {
  capabilities = ["read"]
}
path "secret/metadata/demo/*" {
  capabilities = ["read", "list"]
}
EOF

echo "==> Write policy demo-read"
vault policy write demo-read "$POLICY_FILE"

echo "==> Create child token with demo-read (30m TTL)"
# TODO(you): change -ttl, try -renewable=true, then vault token renew / revoke
# — see TASKS.md §1 and §2b ✅
CHILD_TOKEN=$(vault token create -policy=demo-read -ttl=30m -field=token)
echo "Child token (keep private): $CHILD_TOKEN"

echo "==> Lookup child token"
VAULT_TOKEN="$CHILD_TOKEN" vault token lookup

echo "==> Read secret as child (should work)"
VAULT_TOKEN="$CHILD_TOKEN" vault kv get -field=username secret/demo/db

echo "==> Write as child (should FAIL — expected)"
if VAULT_TOKEN="$CHILD_TOKEN" vault kv put secret/demo/db username=hacked password=nope; then
  echo "ERROR: child was able to write — tighten the policy"
  exit 1
else
  echo "Good: permission denied on write."
fi

echo ""
echo "Next:"
echo "  - TASKS.md §2 (seal/unseal, leases, namespaces theory)"
echo "  - TODO(you): create your own policy file under local/policies/ and use it"
echo "  - Try: VAULT_TOKEN=$CHILD_TOKEN vault token renew"
echo "  - Try: vault token revoke $CHILD_TOKEN"
