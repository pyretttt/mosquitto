# db_management — Master Task List

Start here. Tasks are grouped by concern and ordered roughly easiest →
hardest within each group. Inline `TODO(you)` markers in the code point back to
these items. Tick them off as you go.

**Time box:** ~2–3 sessions × 2–3 hours. Suggested order: §0 → §1 → §2 → §3 →
§4 → §5 → §6 optional.

Legend: `[ ]` todo · `[~]` partially scaffolded · `[x]` done

Pair each section with `docs/CONCEPTS.md`.

---

## Problem

You run a small **records API** backed by **PostgreSQL** on a local kind
cluster. Product needs HTTPS on a stable hostname, ops needs a backup/restore
path that does not require your laptop’s cluster-admin kubeconfig forever, and
security wants the database unreachable from random pods.

**Done looks like:** `https://records.localtest.me/records` works with your
self-signed CA; you can dump and restore the DB (host helper first, then an
in-cluster Job under `pg-backup` ServiceAccount); default-deny NetworkPolicies
still allow Ingress→API and API→DB; and you have changed at least two
ingress-nginx settings (ConfigMap + annotation) with a verify step for each.

---

## Learning outcomes

- Deploy Postgres + FastAPI on kind via Helm / mise
- Terminate TLS at Ingress with a self-signed cert
- Backup and restore Postgres; automate with a Job + SA
- Enforce least-privilege NetworkPolicies
- Tune ingress-nginx ConfigMap and Ingress annotations

---

## 0. Bootstrap

- [~] Copy `.env.example` → `.env` and set a lab `POSTGRES_PASSWORD`.
- [ ] Install mise if needed; from `ops/db_management` run `mise install`.
- [ ] `mise run verify-tools` (kubectl, helm, kind, docker, openssl).
- [ ] `mise run cluster-up` then `mise run ns`.
- [ ] `mise run ingress-install` and wait until the controller is Available.
- [ ] **Verify:** `kubectl get nodes` and
      `kubectl -n ingress-nginx get pods` show Ready.

---

## 1. Postgres + records API

Bring up data and the app *before* TLS so failures are easier to see.

- [~] **Task:** `mise run deploy-postgres`. Inspect the Bitnami Secret keys:
      ```bash
      kubectl -n records get secret records-pg-postgresql -o jsonpath='{.data}' | jq 'keys'
      ```
      Confirm `charts/records-api` password key matches (script auto-detects
      common names).
- [~] **Task:** `mise run build-load` then `mise run deploy-app`.
- [ ] **Task:** `mise run port-forward` and hit the API:
      ```bash
      curl -s localhost:8080/health
      curl -s localhost:8080/records
      curl -s -X POST localhost:8080/records \
        -H 'content-type: application/json' \
        -d '{"name":"delta","note":"before backup"}'
      ```
- [ ] **Task (optional):** replace the startup seed in `app/main.py` with an
      init Job or migration — leave a note in `docs/notes.md`.
- [ ] **Verify:** `/records` returns at least the seeded rows plus your POST.

---

## 2. Ingress with TLS termination

Self-signed cert, Secret, Ingress object. See `docs/CONCEPTS.md` §1.

- [~] **Task:** `mise run gen-certs` (writes `certs/` and Secret `records-tls`).
- [ ] **Task:** copy `k8s/ingress.yaml.example` → `k8s/ingress.yaml`, adjust
      host if you changed `INGRESS_HOST`, then `mise run apply-ingress`.
- [ ] **Task:** confirm the Ingress address / CLASS:
      ```bash
      kubectl -n records get ingress records-api -o wide
      ```
- [ ] **Verify:**
      ```bash
      curl -s --cacert certs/ca.crt https://records.localtest.me/health
      curl -s --cacert certs/ca.crt https://records.localtest.me/records
      ```
      Without `--cacert` (or `-k`) you should see a TLS trust error — expected
      for self-signed.

---

## 3. Backup and restore Postgres

First use the host helper, then graduate to an in-cluster Job + SA.

- [~] **Task:** `mise run backup-pg`. Confirm a file under `backups/*.dump`.
- [ ] **Task:** insert another row via the API, then restore the *previous*
      dump:
      ```bash
      mise run restore-pg FILE=backups/<your-file>.dump
      ```
      **Verify:** the post-backup row is gone (or back to dump contents); API
      still healthy.
- [ ] **Task:** copy `k8s/rbac/backup-sa.yaml.example` →
      `k8s/rbac/backup-sa.yaml`, tighten the Role rules to what you need, apply
      with `mise run apply-backup-rbac`.
- [ ] **Task:** copy `k8s/backup/pg-dump-job.yaml.example` → a real Job
      manifest; implement `pg_dump -Fc` to the PVC; set
      `serviceAccountName: pg-backup`. Apply and wait for Completed.
- [ ] **Verify:** `kubectl -n records logs job/pg-dump-once` (or your Job name)
      and list files on the backup PVC (exec a debug pod that mounts it).
- [ ] **Stretch:** convert the Job into a `CronJob` (schedule every hour for the
      lab) and document how you would offload dumps to object storage.

---

## 4. NetworkPolicies

Break open access, then reopen the minimum paths. See `docs/CONCEPTS.md` §3.

**Note:** Postgres pods need the label `db-lab.example/tier: database` (set in
`charts/postgresql/values-local.yaml`). API pods already have
`db-lab.example/tier: api`.

- [ ] **Task:** confirm labels:
      ```bash
      kubectl -n records get pods -L db-lab.example/tier
      ```
      If the DB pod lacks the label, restart/upgrade Postgres with the values
      file or `kubectl label` it for the lab.
- [ ] **Task:** copy the four `k8s/networkpolicy/*.yaml.example` files to
      `*.yaml` (start with default-deny + DNS, then API↔DB, then Ingress→API).
      Apply with `mise run apply-netpol` (or one file at a time).
- [ ] **Break-then-fix:** after default-deny alone, show that
      `/health` fails (port-forward or Ingress). After allowing DNS + API→DB +
      Ingress→API, show it works again.
- [ ] **Verify:** from a throwaway pod *without* the api label, `pg_isready` /
      TCP to `records-pg-postgresql:5432` should fail; from the API pod it
      succeeds.
- [ ] **Task:** ensure the backup Job still works — you may need a policy
      allowing `db-lab.example/tier: backup` → database:5432.

---

## 5. nginx configuration

Controller-wide ConfigMap vs per-Ingress annotations. See `docs/CONCEPTS.md` §5.

- [ ] **Task:** find the live controller ConfigMap:
      ```bash
      kubectl -n ingress-nginx get cm
      ```
      Copy ideas from `k8s/nginx/ingress-controller-configmap.yaml.example`,
      patch the real ConfigMap (do not blindly apply a second object with the
      wrong name).
- [ ] **Verify:** change `proxy-body-size` to `1m`, then try a POST body larger
      than 1m (expect 413). Raise it and retry.
- [ ] **Task:** add an Ingress annotation (e.g. rate limit or
      `nginx.ingress.kubernetes.io/configuration-snippet` *only if* snippets are
      enabled — prefer documented annotations like
      `nginx.ingress.kubernetes.io/proxy-connect-timeout`).
- [ ] **Verify:** `kubectl -n ingress-nginx logs deploy/ingress-nginx-controller`
      shows a reload; your annotation affects only `records-api`.

---

## 6. ServiceAccounts hardening (optional)

If §3 used a wide Role, tighten it now.

- [ ] **Task:** remove unused verbs/resources from `pg-backup` Role. Prefer
      mounting the DB Secret into the Job over `pods/exec`.
- [ ] **Task:** create a second SA that can *only* `get` the backup PVC / read
      dumps (restore operator) and prove the backup SA cannot delete unrelated
      Deployments.
- [ ] **Verify:**
      ```bash
      kubectl -n records auth can-i create jobs --as=system:serviceaccount:records:pg-backup
      kubectl -n records auth can-i delete deployments --as=system:serviceaccount:records:pg-backup
      ```
      Expect yes / no respectively.

---

## Notes

Capture commands, dump filenames, and policy gotchas in `docs/notes.md`
(create when you start).
