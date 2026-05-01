#!/usr/bin/env bash

set -euo pipefail

# TODO(you) #1 — secrets on the command line.
# Passing the token as $1 means it lands in shell history and `ps` output.
# Read it from stdin or from an env var instead, e.g.:
#   GITLAB_RUNNER_TOKEN=... ./register-runner.sh
# and then `read -rs TOKEN` if not in the env. Bonus: validate format. ✅

# Inside the compose network, GitLab is reachable as http://gitlab/.
# From the runner's POV that hostname resolves via Docker's DNS.
GITLAB_URL="${GITLAB_URL:-https://gitlab.com}"
GITLAB_RUNNER_TOKEN="${GITLAB_RUNNER_TOKEN:-}"

if [ -z "$GITLAB_RUNNER_TOKEN" ]; then
  printf 'GITLAB_RUNNER_TOKEN is required.\n' >&2
  printf 'Usage: GITLAB_RUNNER_TOKEN=... GITLAB_URL=https://gitlab.com ./scripts/register-runner.sh\n' >&2
  exit 1
fi

# NOTE: no more --docker-network-mode.
# Job containers are now spawned by the sibling `gitlab-runner-dind` daemon
# (see docker-compose.yml), which has its own bridge network and cannot see
# the host's `mlops_mlops`. Pinning a host-side network name here would just
# fail at job-create time. Jobs reach gitlab.com via the sibling's default
# bridge — that's all the current pipelines need. If a future job needs to
# reach mlflow / ml-app directly, expose them on host ports and use
# host.docker.internal, or move to the kubernetes executor (docs/K8S_MIGRATION.md).

# TODO(you) #2 — not idempotent.
# Re-running this command registers a *second* runner with the same name.
# Either (a) detect existing registration and skip, or (b) `gitlab-runner unregister --all-runners` first.
# Bonus: store the resulting runner token in a docker volume so re-registration after a wipe is trivial. ✅

docker compose exec -T gitlab-runner gitlab-runner unregister --all-runners --no-interactive
docker compose exec -T gitlab-runner \
  gitlab-runner register \
    --non-interactive \
    --url "$GITLAB_URL" \
    --token "$GITLAB_RUNNER_TOKEN" \
    --executor docker \
    --docker-image "gcr.io/kaniko-project/executor:debug" \
    --description "local-kaniko-runner" \
    --tag-list "kaniko,local,docker" \
    --run-untagged="true" \
    --locked="false"

# TODO(you) #3 — verification.
# The command above can succeed even when the runner can't actually reach
# GitLab (wrong URL, firewall, etc.). After registration, poll
# `docker compose exec gitlab-runner gitlab-runner verify` and assert all
# runners are alive. Fail the script if any are not. ✅
docker compose exec -T gitlab-runner gitlab-runner verify --no-interactive


echo "Runner registered. Verify in GitLab Admin Area -> Runners."

# TODO(you) #4 (stretch) — security. ✅
# The runner container no longer mounts /var/run/docker.sock; it talks to
# `gitlab-runner-dind` over TCP+TLS, so a compromised job cannot reach the
# host daemon. The sibling daemon is still privileged (any nested dockerd is)
# but its blast radius is one container + its volumes.
# Image builds use Kaniko, not docker:dind — see ml-app/.gitlab-ci.yml.
