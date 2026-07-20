# Concepts — Kubernetes secret management

Short theory companion to `TASKS.md`. Vault is intentionally out of scope
(see `ops/vault_proj`).

## 1. ConfigMap vs Secret

| | ConfigMap | Secret |
| - | --------- | ------ |
| Intended data | Non-sensitive config | Credentials, tokens, certs |
| API storage | UTF-8 strings in `.data` | Base64 in `.data` (or `stringData` helper) |
| Default etcd | Plaintext | Plaintext (base64 is **encoding**, not encryption) |
| RBAC | Separate resource | Separate resource — often tighter Role bindings |
| Mount options | env / volume | env / volume / projected |

**Base64** exists so binary values (certs, keystores) survive YAML/JSON. Anyone
with `get secrets` can decode instantly:

```bash
kubectl get secret demo -o jsonpath='{.data.API_TOKEN}' | base64 -d
```

So: Secret ≠ encrypted. It is a typed API object with different defaults and
conventions — still visible to the API server, etcd (unless encrypted), and
anyone with sufficient RBAC.

## 2. Encryption at rest

Kubernetes can encrypt **etcd values** for selected resources via an
`EncryptionConfiguration` on kube-apiserver (`--encryption-provider-config`).

Common providers:

- `identity` — no encryption (default)
- `aescbc` / `aesgcm` — local keys (lab-friendly; key management is your problem)
- `kms` / `kmssvc` — envelope encryption via an external KMS (production shape)

What it fixes: disk/backup leakage of etcd. What it does **not** fix: anyone
with API access still reads Secrets via kubectl; git still must not hold
plaintext; no rotation product by itself.

Migration: put the new provider first, keep `identity` as fallback, then run
`kubectl get secrets --all-namespaces -o json | kubectl replace -f -` so existing
objects are rewritten encrypted.

## 3. Git-safe secrets: three popular patterns

### SOPS (Secrets OPerationS)

Encrypt file contents (YAML/JSON/env) with age, PGP, or cloud KMS. Commit the
ciphertext; decrypt in CI or locally with the private key / KMS IAM.

- **Pros:** works with any GitOps tool; no cluster controller required to *store*
- **Cons:** you still need a secure decrypt path; key distribution is the hard part

### Sealed Secrets

Controller in-cluster holds a private key. `kubeseal` encrypts asymmetrically
so only that cluster can decrypt. You commit a `SealedSecret` CR; the controller
writes a normal `Secret`.

- **Pros:** simple GitOps story; cluster-scoped trust
- **Cons:** sealed blob is cluster-specific (or needs key backup/restore); not ideal for multi-cluster without care

### External Secrets Operator (ESO)

`ExternalSecret` CR syncs from an external store (AWS SM, GCP SM, Azure KV,
Kubernetes, Fake, …) into a native Secret.

- **Pros:** single source of truth outside the cluster; refresh/rotation hooks
- **Cons:** another operator + store credentials / IRSA / WI to manage

| Pattern | Ciphertext lives | Decrypt happens | Best when |
| ------- | ---------------- | --------------- | --------- |
| SOPS | Git | CI / human / hook | Multi-env files, KMS already exist |
| Sealed Secrets | Git (`SealedSecret`) | In-cluster controller | One cluster, GitOps, no external SM |
| ESO | External store | Operator sync | Central secret manager / cloud |

**Not covered here:** HashiCorp Vault (see `vault_proj`), Secrets Store CSI
Driver (mount cloud secrets as volumes — optional hardening).

## 4. Injection shapes

- **env from Secret** — simple; value visible in process env / crash dumps
- **volume mount** — file on disk; often preferred for certs
- **projected volumes** — combine SA token + config + secret

Pods do not auto-reload when a Secret changes unless you restart or use a
reloader sidecar/controller. This lab bumps an annotation / helm values when
you switch sources.

## 5. Other topics worth knowing

- **RBAC:** `get/list/watch` on secrets is powerful — prefer Role per namespace
- **Immutable Secrets:** `immutable: true` prevents updates (forces new object)
- **ServiceAccount tokens:** projected, time-bound tokens ≠ Opaque app secrets
- **Image layers:** never `ENV PASSWORD=...` or `COPY secrets/` into images
- **CI:** short-lived OIDC → cloud KMS / SM beats long-lived bot tokens in variables
