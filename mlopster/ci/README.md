# CI templates

Thin, reusable GitLab CI files `include`d by the root `.gitlab-ci.yml`.
Designed for the **free tier** (shared SaaS runners, unprivileged).

| File | Stage | What it does |
| --- | --- | --- |
| `lint.gitlab-ci.yml`   | lint   | `ruff` on the app + `helm lint`/`helm template` on charts |
| `test.gitlab-ci.yml`   | test   | `pytest` for the model server (JUnit report) |
| `build.gitlab-ci.yml`  | build  | Kaniko build + push of the model-server image (default branch / tags) |
| `deploy.gitlab-ci.yml` | deploy | manual `helm upgrade` (dry-run until a cluster is reachable) |

## Notes for free tier

- **Kaniko, not dind:** shared runners are unprivileged. Kaniko builds without
  Docker, no `privileged = true` needed.
- **Minutes:** MRs run lint + test only; image builds are gated to the default
  branch + tags. pip is cached per ref.
- **Registry:** images go to the project's built-in Container Registry via the
  CI job token (`$CI_REGISTRY*` are provided automatically). Update
  `model-server` chart `image.repository` to match.
- **Secrets:** add `KUBE_CONFIG_B64` (and any registry creds) as **masked,
  protected** CI/CD variables — never commit them.

See `TASKS.md` §4 for the CI checklist.
