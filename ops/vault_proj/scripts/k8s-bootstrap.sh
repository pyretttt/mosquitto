#!/usr/bin/env bash
# Bootstrap in-cluster Vault: KV sample secret + Kubernetes auth for demo-app.
# Run after: mise run vault-install, and Vault is unsealed / ready.
#
# Typical port-forward (adjust svc name):
#   kubectl -n vault port-forward svc/vault 8200:8200
#   export VAULT_ADDR=http://127.0.0.1:8200
#   export VAULT_TOKEN=<root or bootstrap token>
set -euo pipefail

: "${VAULT_ADDR:?Set VAULT_ADDR to reachable Vault API}"
: "${VAULT_TOKEN:?Set VAULT_TOKEN}"
NS_APP="${NS_APP:-demo}"
SA_NAME="${SA_NAME:-demo-app}"

echo "==> Enable KV v2"
vault secrets enable -path=secret kv-v2 2>/dev/null || true

echo "==> Write app secret"
vault kv put secret/demo/db username=app password="k8s-lab-only-rotate-me"

POLICY_FILE="$(mktemp)"
trap 'rm -f "$POLICY_FILE"' EXIT
cat >"$POLICY_FILE" <<'EOF'
path "secret/data/demo/*" {
  capabilities = ["read"]
}
EOF
vault policy write demo-app "$POLICY_FILE"

echo "==> Enable Kubernetes auth"
vault auth enable kubernetes 2>/dev/null || true

# TODO(you): complete Kubernetes auth configuration — see TASKS.md §3
# Hints:
#   1. Get the SA JWT issuer / JWKS or use the legacy service account token review path.
#   2. vault write auth/kubernetes/config \
#        kubernetes_host=https://kubernetes.default.svc:443 \
#        token_reviewer_jwt=... \
#        kubernetes_ca_cert=...
#   3. vault write auth/kubernetes/role/demo-app \
#        bound_service_account_names=$SA_NAME \
#        bound_service_account_namespaces=$NS_APP \
#        policies=demo-app \
#        ttl=1h
#
# Modern Vault + K8s often use:
#   vault write auth/kubernetes/config kubernetes_host="https://$KUBERNETES_PORT_443_TCP_ADDR:443"
# when configuring FROM a pod; from your laptop you need the cluster API URL + CA.
#
# Alternative for beginners: AppRole instead of K8s auth — still fill role_id/secret_id
# into a K8s Secret (not committed) and teach rotation of secret_id.

echo ""
echo "Scaffold stopped at TODO(you): wire auth/kubernetes/config + role/demo-app"
echo "Namespace/SA expected: $NS_APP / $SA_NAME"
echo "Then restart demo-app and curl /secret"

# TODO(you): optional PKI bootstrap — see TASKS.md §3d
# vault secrets enable pki
# vault secrets tune -max-lease-ttl=87600h pki
# vault write -field=certificate pki/root/generate/internal common_name="Vault Lab CA" ttl=87600h
# vault write pki/config/urls issuing_certificates="$VAULT_ADDR/v1/pki/ca" crl_distribution_points="$VAULT_ADDR/v1/pki/crl"
# vault write pki/roles/demo-app allowed_domains="demo-app,demo-app.demo,demo-app.demo.svc" allow_subdomains=true max_ttl=72h
