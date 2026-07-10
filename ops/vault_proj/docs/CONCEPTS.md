# Vault concepts (theory track)

Read alongside `TASKS.md`. This is the “why / what” companion to the CLI and
Kubernetes exercises — not a replacement for the [official Vault docs](https://developer.hashicorp.com/vault/docs).

---

## 1. The problem Vault solves

Secrets sprawl: passwords in git, long-lived cloud keys in CI variables, TLS
keys on disk forever, “who has the prod DB password?” answered by tribal knowledge.

Vault centralizes:

1. **Identity** — who/what is asking (human, CI job, pod).
2. **Policy** — which paths and operations are allowed.
3. **Secrets engines** — where secrets live or how they are *generated*.
4. **Lease / TTL** — how long a secret or token remains valid.
5. **Audit** — request/response trail for compliance and incident response.

You still need good ops habits (unseal key ceremony, HA, backups). Vault is not
magic — it is a well-designed control plane for secrets.

---

## 2. Architecture in one page

```
┌─────────────┐     auth      ┌──────────────────────────────────────┐
│ CLI / App / │ ────────────► │              Vault                   │
│ CI / Agent  │ ◄──────────── │  Auth methods → Token + policies     │
└─────────────┘   token/lease │  Secrets engines (KV, PKI, DB, …)    │
                              │  Storage backend (raft, consul, …)   │
                              │  Seal (master key / auto-unseal)     │
                              └──────────────────────────────────────┘
```

- **Barrier** — Vault encrypts data before writing to storage. Storage compromise
  without unseal keys should not yield plaintext secrets.
- **Seal** — on startup (or after `operator seal`), Vault refuses to decrypt.
  **Unseal** reconstructs the master key (Shamir shares or cloud KMS auto-unseal).
- **Dev mode** — in-memory, auto-unsealed, root token printed once. Lab only.

---

## 3. Seal & unseal

| Mode | Behavior |
| ---- | -------- |
| Shamir (default) | `init` creates key shares; threshold (e.g. 3 of 5) to unseal |
| Auto-unseal | Cloud KMS / HSM unwraps the master key; no human key entry |
| Sealed | Almost all secrets APIs return errors; status still works |

**Why seal?** Stolen storage + stolen process memory scenarios; controlled
restart; break-glass when you suspect compromise (`vault operator seal`).

**Lab tip:** `-dev` hides this complexity. Use `mise run vault-file` once so you
feel `init` + multiple `unseal` commands.

---

## 4. Auth methods & tokens

Authentication answers: “who are you?” Vault then issues a **token** bound to
**policies**.

Common auth methods:

| Method | Typical use |
| ------ | ----------- |
| Token | Break-glass, initial root, humans in labs |
| Userpass / OIDC | Humans |
| AppRole | Machines / CI without K8s |
| Kubernetes | Pods use their ServiceAccount JWT |
| Cloud (AWS/GCP/Azure) | Cloud-native workloads |

### Tokens

- Carry **policies** (ACL rules on paths like `secret/data/demo/*`).
- Have **TTL** / **max TTL**; may be **renewable** or **periodic**.
- Can be **revoked** individually or as a tree (`revoke -self`, orphan tokens, …).
- Root token = superpower; stop using it after bootstrap.

Policy sketch:

```hcl
path "secret/data/demo/*" {
  capabilities = ["read"]
}
```

Note KV v2 API paths: data lives under `secret/data/...`; metadata under
`secret/metadata/...`.

---

## 5. Leases — renew & revoke

Many Vault responses include a **lease_id** and **lease_duration**:

- Dynamic DB credentials, PKI certs (depending on config), tokens themselves.
- **Renew** extends life up to `max_ttl` / role limits.
- **Revoke** invalidates immediately (and may revoke the backing credential).

Static KV v2 secrets are a bit different: versions are not classic leases, but
you still design **rotation** (new version, consumers refresh, old version
destroyed or left for rollback).

Mental model: prefer **short TTL + renew** over immortal passwords.

---

## 6. Namespaces (lock & unlock)

**Namespaces** (Vault Enterprise / HCL Vault Dedicated) are hierarchical
isolation: separate policies, auth methods, and mounts per team or tenant.

- Admins can **lock** a namespace — API calls inside it fail until **unlock**.
- Useful for: compromised tenant, billing freeze, maintenance.

**Open source substitute for this lab:** separate KV mount paths + policies per
“team”. You do not get true namespace lock, but you practice isolation and least
privilege — the skill that transfers.

---

## 7. Secrets engines you will touch

| Engine | Role in this project |
| ------ | -------------------- |
| **KV v2** | Versioned key/value (DB username/password) |
| **PKI** | Private CA — issue TLS for the demo app without Let’s Encrypt / paid CA |
| **Transit** (optional) | Encrypt/decrypt without exposing keys (“encryption as a service”) |
| **Database** (optional) | Dynamic DB users with leases |

### PKI without a paid CA

Vault acts as your **internal CA**:

1. Enable `pki`, generate root (lab) or root→intermediate (better).
2. Define a role (allowed domains, TTL).
3. Issue cert/key; distribute via K8s Secret, Agent, or CSI.
4. Short TTLs + renew/reissue = rotation for free.

Browsers will not trust your private CA unless you install the CA cert — for a
lab, `curl -k` or trust the CA in your OS/trust store.

---

## 8. Kubernetes integration patterns

Three common patterns (pick one in TASKS §3):

1. **Vault Agent Injector** — sidecars render secrets to files/env from annotations.
2. **App authenticates itself** — code uses K8s auth + Vault API (what `app/main.py`
   scaffolds).
3. **Secrets Operator / CSI** — sync Vault → native K8s Secret objects.

For beginners, (2) makes the auth flow visible; (1) is closer to many prod setups.

---

## 9. Expiration & rotation (ops habits)

| Technique | Idea |
| --------- | ---- |
| Short token TTL | Stolen token dies quickly |
| Renewable leases | Healthy apps renew; dead apps expire |
| KV versions | Write v2, cut over, delete v1 |
| Dynamic secrets | Vault creates DB user; revoke drops user |
| PKI short TTL | Certs re-issued often |

Rotation runbook (manual lab version) is an exercise in TASKS §4.

---

## 10. What not to do

- Commit unseal keys, root tokens, or `.env` with real credentials.
- Use `-dev` or UI “dev server” in production.
- Give every pod the root token.
- Store forever-passwords and call it “done” because they live in Vault.

When you finish TASKS.md, you should be able to teach a teammate the diagram in
§2 and demo seal, a limited token, a lease revoke, and an app reading KV on K8s.
