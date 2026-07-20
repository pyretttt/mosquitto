# secret_management — Master Task List

Start here. Tasks are grouped by concern and ordered roughly easiest → hardest
within each group. Inline `TODO(you)` markers in the code point back to these
items. Tick them off as you go.

**Time box:** ~2–3 days × 2–3 hours. Suggested order: §0 → §1 → §2 → pick §3
*or* §4 first → §5 → §6 optional.

Legend: `[ ]` todo · `[~]` partially scaffolded · `[x]` done

Pair each section with `docs/CONCEPTS.md`.

---

## Problem

Your team runs apps on Kubernetes. Non-sensitive settings live in ConfigMaps;
credentials were pasted into YAML and committed, or stuffed into Secrets with
the assumption that “Kubernetes encrypts them.” They do not — by default Secrets
are base64-encoded in etcd, and anyone with `get secrets` can read them. You
also need a story for **git**: how to review and deploy secret-backed config
without plaintext in the repo.

**Done looks like:** you can explain ConfigMap vs Secret and encryption-at-rest
out loud; demonstrate SOPS, Sealed Secrets, and External Secrets on a local
cluster; and run the demo app against at least two different Secret sources
without leaking raw tokens in `/secret-status`.

---

## Learning outcomes

- Explain ConfigMap vs Secret and why base64 ≠ encryption
- Enable and verify encryption-at-rest for Secrets on kind
- Use SOPS + age to encrypt Secret manifests in git
- Use Sealed Secrets (`kubeseal`) for cluster-scoped ciphertext in git
- Use External Secrets Operator to sync into native Secrets (no Vault)
- Switch the demo app between Secret sources and verify safely

---

## 0. Bootstrap

- [x] Copy `.env.example` → `.env` and set `DEMO_API_TOKEN` to a lab-only value.
- [x] `mise install` from `ops/secret_management`.
- [x] **Verify:** `kubectl version --client`, `helm version`, `kind version`,
      `sops --version`, `age --version`, `kubeseal --version`.
- [x] Create a cluster: `mise run cluster-kind` (plain cluster for §1; you may
      recreate with encryption in §2).
- [x] `mise run ns` and `mise run configmap`.
- [x] Skim `README.md` and `docs/CONCEPTS.md` §1.

---

## 1. ConfigMap vs Secret (base64 is not encryption)

Kubernetes Secrets feel “safer” than ConfigMaps, but the API stores Secret
`.data` as **base64**. That is encoding for transport/binary safety — not
confidentiality.

- [x] **Task:** read `k8s/01-configmap.yaml` and `k8s/02-secret.yaml.example`.
- [x] **Task:** create the Opaque Secret: export `DEMO_API_TOKEN` then
      `mise run secret-create` (or finish the example file as
      `*.plain.yaml` and apply — keep plaintext out of git).
- [x] **Task:** run `mise run inspect-cm-secret`. Decode the Secret field by
      hand with `base64 -d`.
- [x] **Task:** build and deploy the app:
      ```bash
      mise run build-load
      mise run deploy-app
      mise run port-forward
      ```
      In another shell: `curl -s localhost:8080/config` and
      `curl -s localhost:8080/secret-status`.
- [x] **Verify:** `/config` shows ConfigMap values; `/secret-status` shows
      `token_present: true` and a **masked** token (not the raw value).
- [x] **Reflect:** who in your RBAC model can `get secrets` in `demo`? Why is
      that more sensitive than `get configmaps`?

---

## 2. Encryption at rest

Without `--encryption-provider-config`, Secret objects in etcd are stored in a
form you can often recognize (YAML/JSON + base64). Encryption-at-rest protects
**disk and etcd backups**, not API readers.

- [x] Scaffold: `k8s/encryption/EncryptionConfiguration.yaml.example`,
      `kind/config.yaml`, `scripts/kind-with-encryption.sh`,
      `scripts/etcd-peek-secret.sh`.
- [x] **Task:** delete the plain cluster if you want a clean slate:
      `kind delete cluster --name secrets-lab`.
- [x] **Task:** `mise run cluster-kind-encrypt` (generates a local
      `EncryptionConfiguration.yaml` with a fresh key — do not commit it).
- [x] Re-apply namespaces, ConfigMap, Secret (`mise run ns configmap secret-create`).
- [x] **Task:** `mise run etcd-peek` and inspect the blob.
- [x] **Verify:** output contains something like `k8s:enc:aescbc:v1:key1:` rather
      than readable Secret YAML. (If you still see plaintext, the apiserver flag
      did not take effect — check kind mounts / recreate cluster.)
- [x] **Optional:** create a Secret *before* enabling encryption on a throwaway
      cluster, peek etcd, then recreate with encryption and compare.
- [ ] Skim CONCEPTS §2. Note what encryption-at-rest does **not** solve.

---

## 3. SOPS + age (encrypt files in git)

SOPS encrypts selected YAML fields so the ciphertext can live in git. Decrypt
with a private age key (lab) or cloud KMS (production shape).

- [~] Scaffold: `.sops.yaml`, `secrets/demo-app.enc.yaml.example`,
      `mise run sops-age-init|sops-encrypt|sops-decrypt-apply`.
- [ ] **Task:** `mise run sops-age-init`. Put the **public** key into
      `.sops.yaml` (`age: age1...`). Keep `local/age/key.txt` private.
- [ ] **Task:** create `secrets/demo-app.plain.yaml` (copy from
      `k8s/02-secret.yaml.example`, rename metadata.name to `demo-app-sops`,
      set real lab values). Ensure `*.plain.yaml` stays gitignored.
- [ ] **Task:** `mise run sops-encrypt` → produces `secrets/demo-app.enc.yaml`.
- [ ] **Verify:** open the `.enc.yaml` — `stringData`/`data` values should be
      `ENC[AES256_GCM,...]` ciphertext. Commit the encrypted file only.
- [ ] **Task:** `mise run sops-decrypt-apply`.
- [ ] Point the chart at it: set `secret.name: demo-app-sops` and
      `secret.sourceHint: sops` in `charts/demo-app/values-local.yaml`, then
      `mise run deploy-app` and curl `/secret-status`.
- [ ] **Verify:** `source_hint` is `sops` and token is present.

---

## 4. Sealed Secrets

Sealed Secrets encrypt for a **specific cluster** (controller private key).
`kubeseal` produces a `SealedSecret` CR safe to commit; the controller writes
the native Secret.

- [~] Scaffold: `sealed-secrets/demo-app-sealed.yaml.example`,
      `mise run sealed-install|sealed-seal|sealed-apply`.
- [ ] **Task:** `mise run sealed-install`. Wait until the controller pod is Ready.
- [ ] **Task:** `mise run sealed-seal` (uses `DEMO_API_TOKEN`).
- [ ] **Task:** `mise run sealed-apply`.
- [ ] **Verify:** `kubectl -n demo get sealedsecret,secret demo-app-sealed`.
- [ ] Point the chart: `secret.name: demo-app-sealed`,
      `sourceHint: sealed-secrets`, redeploy, curl `/secret-status`.
- [ ] **Reflect:** what happens if you restore this `SealedSecret` onto a
      *different* cluster without restoring the sealing key?

---

## 5. External Secrets Operator (no Vault)

ESO syncs from an external store into a native Secret. This lab uses the
**Fake** provider first (no cloud account). Vault is deliberately avoided —
you already covered it in `vault_proj`.

- [~] Scaffold: `external-secrets/fake-store.yaml`,
      `external-secrets/kubernetes-store.yaml.example`,
      `mise run eso-install|eso-fake-apply`.
- [ ] **Task:** `mise run eso-install`. Wait for CRDs and controller Ready.
- [ ] **Task:** edit Fake values in `external-secrets/fake-store.yaml` if you
      want distinct tokens, then `mise run eso-fake-apply`.
- [ ] **Verify:** `kubectl -n demo get externalsecret demo-app-eso` shows
      `SecretSynced` / Ready; `kubectl -n demo get secret demo-app-eso` exists.
- [ ] Point the chart: `secret.name: demo-app-eso`, `sourceHint: external-secrets`,
      redeploy, curl `/secret-status`.
- [ ] **Optional:** implement the Kubernetes provider path
      (`kubernetes-store.yaml.example`): create a source Secret in another
      namespace + RBAC for ESO’s ServiceAccount, then sync into `demo`.
- [ ] Skim CONCEPTS §3 table — when would you pick SOPS vs Sealed vs ESO?

---

## 6. Hardening & extras (optional)

- [ ] **RBAC:** create a Role that can `get/list` ConfigMaps but **not** Secrets;
      bind a test ServiceAccount; show `kubectl auth can-i`.
- [ ] **Immutable Secret:** set `immutable: true` on a Secret; try to patch it;
      observe the failure; create a new object instead.
- [ ] **Volume mount vs env:** change the Deployment to mount the Secret as a
      file under `/var/run/secrets/demo` and read it from the app (small code
      change) — CONCEPTS §4.
- [ ] **Secret change / restart:** update a Secret value and note the pod still
      has the old env until restart; add a reloader annotation or bump a
      checksum annotation via Helm.
- [ ] **CSI (read-only):** skim Secrets Store CSI Driver docs; write 5 lines in
      CONCEPTS on how it differs from ESO (mount vs synced Secret object).
- [ ] **Never in images:** grep your Dockerfile history mentally — confirm no
      secret `ENV` / `COPY` of plaintext credentials.

---

## Suggested day plan

| Day | Focus |
| --- | ----- |
| 1 | §0 Bootstrap, §1 CM vs Secret, start §2 encryption-at-rest |
| 2 | Finish §2, §3 SOPS **or** §4 Sealed Secrets, redeploy app |
| 3 | The other of §3/§4, §5 ESO, optional §6 |

When finished, you should be able to answer: *Where does ciphertext live, who
can decrypt, and what still trusts the Kubernetes API?*
