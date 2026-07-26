# Concepts — DB management on Kubernetes

Short theory companion to `TASKS.md`.

## 1. Why Ingress + TLS termination

Clients talk HTTPS to a stable hostname. The **Ingress controller** (here
`ingress-nginx`) terminates TLS using a Secret of type `kubernetes.io/tls`,
then proxies HTTP to your ClusterIP Service.

```
browser --HTTPS--> ingress-nginx --HTTP--> records-api --TCP:5432--> postgres
```

Self-signed certs are fine for labs: generate with openssl, trust via
`curl --cacert`. In production you use a real CA (or cert-manager + ACME).

**`localtest.me`** (and `*.localtest.me`) resolves to `127.0.0.1`, so kind’s
hostPort 443 mapping works without editing `/etc/hosts`.

## 2. Postgres backup / restore

| Tool | Use |
| ---- | --- |
| `pg_dump -Fc` | Custom-format dump (flexible restore) |
| `pg_restore` | Load custom-format; `--clean --if-exists` for lab resets |
| PVC / object storage | Where dumps live; Jobs write here |
| CronJob | Schedule; Job = one-shot |

Blast radius: a dump contains **all row data**. Treat backup volumes like
Secrets. Prefer a dedicated ServiceAccount over using your personal kubeconfig
admin powers inside automation.

## 3. NetworkPolicies

Default Kubernetes networking is **allow-all** within the cluster. A
`NetworkPolicy` with an empty `podSelector` and `policyTypes: [Ingress, Egress]`
is a common **default-deny** pattern. Then you open only what you need:

- DNS (kube-dns / CoreDNS) so Services resolve
- API → DB on 5432
- ingress-nginx → API on 8080

CNI must support NetworkPolicy (kindnet does). Wrong labels = silent drops —
use `kubectl describe networkpolicy` and pod labels together when debugging.

## 4. ServiceAccounts & RBAC for backup

Pods authenticate as a **ServiceAccount**. RBAC (`Role` / `RoleBinding`)
grants that identity verbs on resources **in a namespace**.

Backup automation should not run as `cluster-admin`. Typical needs:

- `get/list` Secrets (DB password) or mount the Secret
- `create` Jobs (or run as a Job itself)
- Optionally `pods/exec` if you choose an exec-based design (prefer in-Job
  `pg_dump` with a mounted Secret instead)

## 5. nginx Ingress ConfigMap vs annotations

| Layer | Scope | Examples |
| ----- | ----- | -------- |
| Controller ConfigMap | All Ingresses on that controller | `proxy-body-size`, log formats |
| Ingress annotations | One Ingress object | `nginx.ingress.kubernetes.io/ssl-redirect`, rate limits |

Change ConfigMap → controller reloads. Change annotation → only that route.
Confirm exact key names against your ingress-nginx version.
