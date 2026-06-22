# Admin & Hardening Tasks (separate track)

> Do the main tracks in `TASKS.md` first. This is the "make it
> production-grade" track: certificates, networking, secrets, RBAC, pod
> security, quotas, backups. Each item is ordered roughly by what you'd tackle
> first when promoting the local scaffold to a real cluster.

## A. TLS / Certificates

- [ ] Install **cert-manager** (`jetstack/cert-manager`, with CRDs).
- [ ] Create a `ClusterIssuer`:
      - local: self-signed or a local CA issuer (so you can practice mTLS).
      - real: `letsencrypt-prod` (HTTP-01 via the ingress, or DNS-01).
- [ ] Add `cert-manager.io/cluster-issuer` annotations + `tls:` blocks to the
      Grafana / model-server / Airflow ingresses (stubs already in values).
- [ ] (Advanced) internal mTLS between services — evaluate a service mesh
      (Linkerd is the lightest) vs cert-manager-issued client certs.
- [ ] Automate cert rotation; alert on certs nearing expiry (there's a
      built-in `cert-manager` mixin / Prometheus rules).

## B. Ingress

- [ ] Install an ingress controller (`ingress-nginx`). Locally, expose it via
      the kind/k3d port mapping (80/443).
- [ ] Give each UI a hostname (`grafana.<domain>`, `airflow.<domain>`,
      `model.<domain>`); use `/etc/hosts` + `*.localdev` locally.
- [ ] Force HTTPS redirects; set HSTS where appropriate.
- [ ] Put auth in front of internal UIs (oauth2-proxy / basic auth) if exposed.

## C. NetworkPolicy

- [ ] Default-deny ingress + egress per namespace.
- [ ] Allow only required flows:
      - `model-server → storage` (MinIO :9000)
      - `orchestration → storage` (MinIO :9000)
      - `monitoring → *` scrape ports (metrics endpoints only)
      - DNS egress to kube-dns.
- [ ] Verify by testing a denied path (exec into a pod, curl something blocked).
- [ ] Note: NetworkPolicy needs a CNI that enforces it (Calico/Cilium); kind's
      default kindnet does NOT — install Calico locally to actually test.

## D. Secrets management

- [ ] Get every plaintext password out of values: MinIO root creds, Grafana
      admin, Airflow webserver key + admin, model-server admin token.
- [ ] Pick a tool:
      - **Sealed Secrets** (Bitnami) — encrypt secrets into git, controller
        decrypts in-cluster. Simplest for a solo repo.
      - **External Secrets Operator** — sync from Vault / cloud secret manager.
      - **SOPS** (+ age/KMS) — file-level encryption, GitOps-friendly.
- [ ] Wire charts to `existingSecret` references (stubs already present).
- [ ] Rotate the S3 keys and confirm all consumers pick up the new Secret.

## E. RBAC & ServiceAccounts

- [ ] Dedicated, least-privilege ServiceAccount per component.
- [ ] `automountServiceAccountToken: false` unless the pod calls the k8s API
      (already set on the model server).
- [ ] Review Airflow's RBAC — KubernetesExecutor needs permission to create
      task pods in its namespace; scope it tightly.
- [ ] Audit `ClusterRole`s pulled in by subcharts (kube-prometheus-stack grants
      broad read).

## F. Pod & cluster security

- [ ] `securityContext`: non-root, `readOnlyRootFilesystem`, drop all caps,
      `allowPrivilegeEscalation: false` (model server done; do the rest).
- [ ] Apply Pod Security Admission labels per namespace
      (`pod-security.kubernetes.io/enforce: restricted` where feasible).
- [ ] `seccompProfile: RuntimeDefault`.
- [ ] Image hygiene: pin digests, scan images in CI (Trivy), no `:latest` in
      prod deploys.

## G. Resource governance & resilience

- [ ] `ResourceQuota` + `LimitRange` per namespace.
- [ ] `PodDisruptionBudget` for Prometheus, Alertmanager, MinIO, Airflow.
- [ ] Requests/limits reviewed on every workload (avoid noisy-neighbor / OOM).
- [ ] Anti-affinity / topology spread for HA on multi-node clusters.

## H. Backups & DR

- [ ] Snapshot PVCs (Prometheus TSDB, Grafana, MinIO, Airflow Postgres) —
      Velero is the usual tool.
- [ ] Back up MinIO buckets (mc mirror to another bucket / region).
- [ ] Document and *test* a restore. A backup you haven't restored isn't one.

## I. Supply chain / GitOps (stretch)

- [ ] Move deploys to Argo CD or Flux (the CI `deploy` stage becomes "render +
      commit", the cluster pulls).
- [ ] Sign images (cosign) and verify in an admission policy (Kyverno / OPA).
