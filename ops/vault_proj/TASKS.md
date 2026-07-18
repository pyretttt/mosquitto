# vault_proj — Master Task List

Start here. Tasks are grouped by track and ordered roughly easiest → hardest
within each group. Inline `TODO(you)` markers in the code point back to these
items. Tick them off as you go.

Legend: `[ ]` todo · `[~]` partially scaffolded · `[x]` done

**Suggested order:** §0 Bootstrap → §1 Local CLI → §2 Core concepts (mix theory
+ CLI) → §3 K8s Vault + demo app → §4 TTL / rotation → §5 Hardening (optional).

Pair each section with the matching chapter in `docs/CONCEPTS.md`.

---

## Problem

Your team ships a tiny backend that needs a database password and HTTPS. Today
those values are either hardcoded, stuffed into a Kubernetes Secret by hand, or
copied from a chat message. There is no expiry, no audit trail, and rotating
the password means a scary outage.

**Done looks like:** you can explain Vault’s model out loud; run Vault on your
laptop and exercise KV, tokens, leases, seal/unseal; deploy Vault + the demo
app on a local cluster so the app obtains secrets (and optionally a TLS cert
from Vault PKI) without baking credentials into the image; demonstrate renew,
revoke, and a simple rotation path.

---

## Learning outcomes

- Explain the problem Vault solves vs env vars / plain K8s Secrets
- Use the Vault CLI for KV, auth, tokens, leases
- Describe seal/unseal, policies, and namespace isolation (incl. lock/unlock)
- Run Vault and a demo app on local Kubernetes
- Issue TLS with Vault PKI (private CA — no paid CA)
- Practice TTL, renew, revoke, and rotation

---

## 0. Bootstrap

- [x] Install tools: `mise install` from `ops/vault_proj`.
- [x] Copy `.env.example` → `.env` and skim the variables (no real secrets needed for `-dev`).
- [x] **Verify:** `vault version`, `kubectl version --client`, `helm version`.
- [x] Skim `README.md` (problem statement + mental model) and `docs/CONCEPTS.md` §1–2.

---

## 1. Local CLI lab (no Kubernetes)

Vault’s `-dev` mode auto-unseals, prints a root token, and stores data in memory.
Perfect for learning the CLI — **never** use `-dev` in production.

- [x] **Task:** start Vault: `mise run vault-dev`. Note the root token and
      `VAULT_ADDR` (usually `http://127.0.0.1:8200`).
- [x] Export in your shell: `export VAULT_ADDR=http://127.0.0.1:8200` and
      `export VAULT_TOKEN=<root-token-from-dev-output>`.
- [x] **Verify:** `vault status` shows `Sealed: false` and `HA Enabled: false`.
- [x] Run the guided script: `mise run local-kv-demo` (or step through
      `scripts/local-kv-demo.sh` yourself).
- [x] **Task:** enable KV v2 yourself if the script did not:
      `vault secrets enable -path=secret kv-v2`
- [x] **Task:** write and read a secret:
      ```bash
      vault kv put secret/demo/db username=app password=s3cret
      vault kv get secret/demo/db
      vault kv get -field=password secret/demo/db
      ```
- [x] **Task:** create a **policy** file that only allows `read` on
      `secret/data/demo/*`, then create a child token with that policy.
      See `TODO(you)` in `scripts/local-kv-demo.sh` and CONCEPTS § Authentication / Tokens.
- [x] **Verify:** with the child token, `vault kv get secret/demo/db` works;
      writing should fail.

---

## 2. Core Vault concepts (theory + practice)

Work through `docs/CONCEPTS.md` while doing the CLI checks below.

### 2a. Seal / unseal

- [x] **Read:** CONCEPTS — Seal & unseal.
- [x] **Task:** with `-dev`, run `vault operator seal`, then `vault status`.
- [x] **Task:** unseal is different in `-dev` (often restart is easiest). Then try
      the **file storage** path: copy `local/vault-config.hcl.example` →
      `local/vault-config.hcl`, run `mise run vault-file`, go through
      `vault operator init` and `vault operator unseal` (3 of 5 keys by default).
- [x] **Verify:** after sealing, API reads fail until unsealed; after unseal,
      `vault status` shows `Sealed: false`.

### 2b. Authentication & tokens

- [x] **Read:** CONCEPTS — Auth methods & tokens.
- [x] **Task:** `vault token lookup` on your root token — note `ttl`, `renewable`, `policies`.
- [x] **Task:** create a periodic or renewable token with your limited policy;
      `vault token renew`; `vault token revoke`.
- [x] **Verify:** after revoke, that token can no longer read secrets.

### 2c. Leases — renew & revoke

- [x] **Read:** CONCEPTS — Leases.
- [x] **Task:** enable the `transit` or use a dynamic-ish path. Easiest beginner
      path: create a token with a short TTL (`-ttl=60s`), watch it expire, and
      practice `vault lease` / `vault token renew` before expiry.
- [x] Optional stretch: enable the **database** secrets engine against a local
      Postgres and issue a dynamic role credential; `vault lease renew` /
      `vault lease revoke` on the lease id.
      Run: `mise run local-db-lease-demo` (or step through
      `scripts/local-db-lease-demo.sh`). Needs Docker + Vault `-dev`.
- [x] **Verify:** after revoke/expiry, the credential no longer works.

### 2d. Namespaces (lock / unlock)

Open-source Vault does **not** include namespaces (Enterprise / HCP feature).
Still learn the model — it shows up in real jobs.

- [x] **Read:** CONCEPTS — Namespaces.
- [x] **Task (theory):** explain in your own words what a namespace isolates and
      what “namespace lock” does to API access inside that namespace.

It isolates API between teams, whole server API can handle request, when namespace can be locked and no request handled in namespace. Teams do not collide with each other.

- [x] **Task (practice substitute on OSS):** use **separate mount paths + policies**
      to simulate two teams (`secret/team-a/...` vs `secret/team-b/...`) so each
      token can only see its path. Document the mapping in a short note under
      `docs/` (optional `docs/namespaces-oss-lab.md`).
- [ ] If you have access to HCP Vault / Enterprise later: practice
      `vault namespace lock` / `unlock` and confirm clients get errors while locked.

---

## 3. Kubernetes: Vault + demo app

Goal: the demo API in `app/` obtains a backend secret from Vault (not from a
hardcoded string). TLS via Vault PKI is in §3d.

- [x] Create a local cluster: `mise run cluster-k3d` (or `cluster-kind`).
- [x] **Verify:** `kubectl get nodes` is Ready.
- [x] Install Vault with the local values overlay:
      `mise run vault-install` (uses `charts/vault-values/values-local.yaml`).
- [x] **Verify:** Vault pod(s) Running; port-forward or use the chart’s service;
      `vault status` against the in-cluster address (see mise task output).
- [x] **Task:** complete `TODO(you)` in `charts/vault-values/values-local.yaml`
      (dev vs standalone, UI, injector/agent notes).
- [x] Enable KV in the cluster Vault and write `secret/demo/db` (same shape as local).
- [x] Configure **Kubernetes auth** (or AppRole) so the demo-app ServiceAccount
      can read that path. Fill `TODO(you)` in `scripts/k8s-bootstrap.sh`.
- [x] Build & load the app image: `mise run app-build`.
- [x] Install the chart: `mise run app-install`.
- [x] **Task:** finish `TODO(you)` in `app/main.py` so `/secret` (or `/health`’s
      dependency) reads from Vault using the pod’s auth method — not a hardcoded password.
- [x] **Verify:** `kubectl port-forward svc/demo-app 8080:8080` then
      `curl -s localhost:8080/health` and `curl -s localhost:8080/secret` show
      success without printing the raw password in logs if you can avoid it.

### 3d. TLS with Vault PKI (no paid CA)

- [ ] **Read:** CONCEPTS — PKI.
- [ ] **Task:** enable `pki` secrets engine; generate a root (or intermediate) CA;
      create a role for `demo-app.local`.
- [ ] **Task:** issue a cert; create a Kubernetes TLS secret (or use Vault Agent /
      CSI — your choice). Wire `charts/demo-app` to serve HTTPS or mount the cert.
      See `TODO(you)` in `charts/demo-app/templates/` and `values-local.yaml`.
- [ ] **Verify:** `curl -k https://…` works; inspect the cert issuer — it should
      be your Vault CA, not a public CA.

---

## 4. Expiration, rotation, renew, revoke

- [ ] Set a short TTL on a KV v2 custom metadata or on a token/dynamic secret
      you use for the demo (`ttl` / `max_ttl` on roles).
- [ ] **Task:** demonstrate **renew** before expiry and **revoke** mid-life;
      show the app or CLI failing after revoke.
- [ ] **Task:** implement a simple **rotation** runbook (manual is fine):
      1) write new password to Vault (new KV version),
      2) reload/restart app or rely on Agent template refresh,
      3) invalidate old version / revoke old lease.
      Capture the steps in `docs/rotation-runbook.md` (create it).
- [ ] **Verify:** old credential stops working; new one works; you can point to
      an audit-friendly sequence of commands.

---

## 5. Hardening (optional)

- [ ] Turn off Vault `-dev` / `devserver` for the K8s install; use file or raft
      storage with init + unseal keys stored safely **outside** git.
- [ ] Add least-privilege policies (app can read only its path).
- [ ] NetworkPolicy: only demo-app can reach Vault on 8200 in-cluster.
- [ ] Enable an audit device (`file` is fine for the lab) and find your `kv get`
      in the log.
- [ ] Replace root token usage with AppRole or K8s auth everywhere in scripts.

---

## Quick reference — mise tasks

| Task | Purpose |
| ---- | ------- |
| `mise run vault-dev` | Local Vault `-dev` on :8200 |
| `mise run vault-file` | Local Vault with file storage (init/unseal practice) |
| `mise run local-kv-demo` | Guided KV + token demo script |
| `mise run local-db-lease-demo` | Postgres + dynamic DB creds / leases |
| `mise run cluster-k3d` / `cluster-kind` | Local Kubernetes |
| `mise run vault-install` | Helm install Vault |
| `mise run app-build` | Build/load demo image |
| `mise run app-install` | Helm install demo-app |
| `mise run down` | Tear down app + vault releases (cluster kept) |
| `mise run destroy-cluster` | Delete kind/k3d cluster |
