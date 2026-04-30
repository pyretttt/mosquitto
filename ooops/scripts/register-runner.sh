#!/usr/bin/env bash

set -euox pipefail

# TODO(you) #1 — secrets on the command line.
# Passing the token as $1 means it lands in shell history and `ps` output.
# Read it from stdin or from an env var instead, e.g.:
#   GITLAB_RUNNER_TOKEN=... ./register-runner.sh
# and then `read -rs TOKEN` if not in the env. Bonus: validate format. ✅

# Inside the compose network, GitLab is reachable as http://gitlab/.
# From the runner's POV that hostname resolves via Docker's DNS.
GITLAB_URL="${GITLAB_URL:-https://gitlab.com}"
GITLAB_RUNNER_TOKEN="${GITLAB_RUNNER_TOKEN:-}"

# TODO(you) #2 — network name is brittle.
# `mlops_mlops` follows the pattern `<compose-project>_<network>`. The
# compose project is set to `mlops` via the top-level `name:` in
# docker-compose.yml. If anyone overrides `COMPOSE_PROJECT_NAME`, the
# `--docker-network-mode` below silently breaks. Compute it dynamically:
#   NETWORK="$(docker network ls --filter 'name=_mlops$' --format '{{.Name}}' | head -n1)" ✅
# and assert it's non-empty.
NETWORK="$(docker network ls --filter 'name=_mlops$' --format '{{.Name}}' | head -n 1)"

# TODO(you) #3 — not idempotent.
# Re-running this command registers a *second* runner with the same name.
# Either (a) detect existing registration and skip, or (b) `gitlab-runner unregister --all-runners` first.
# Bonus: store the resulting runner token in a docker volume so re-registration after a wipe is trivial.

docker compose exec -T gitlab-runner unregister --all-runners --no-interactive
docker compose exec -T gitlab-runner \
  gitlab-runner register \
    --non-interactive \
    --url "$GITLAB_URL" \
    --registration-token "$TOKEN" \
    --executor docker \
    --docker-image "docker:27" \
    --docker-privileged \
    --docker-volumes "/var/run/docker.sock:/var/run/docker.sock" \
    --docker-network-mode "${NETWORK}" \
    --description "local-docker-runner" \
    --tag-list "docker,local" \
    --run-untagged="true" \
    --locked="false"

# TODO(you) #4 — verification.
# The command above can succeed even when the runner can't actually reach
# GitLab (wrong URL, firewall, etc.). After registration, poll
# `docker compose exec gitlab-runner gitlab-runner verify` and assert all
# runners are alive. Fail the script if any are not. ✅
docker compose exec -T gitlab-runner gitlab-runner verify --no-interactive


echo "Runner registered. Verify in GitLab Admin Area -> Runners."

# TODO(you) #5 (stretch) — security.
# `--docker-privileged` + bind-mounting the host docker socket means any
# pipeline job can root the host. Fine for local-only learning, never for
# anything you push to. When you migrate to k8s (next week) the executor
# becomes `kubernetes` and each job runs as its own Pod — much safer.
# Read the trade-offs: https://docs.gitlab.com/runner/executors/docker.html#use-docker-in-docker
