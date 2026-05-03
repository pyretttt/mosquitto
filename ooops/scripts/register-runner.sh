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

# Job containers are spawned by the sibling `gitlab-runner-dind` daemon (see
# docker-compose.yml). In that nested daemon, `host-gateway` points at the
# inner Docker bridge, not Docker Desktop's host gateway, so `host.docker.internal`
# would resolve to a dead end from CI jobs.
#
# `--docker-network-mode host` makes job containers share the DinD container's
# network namespace. That lets jobs resolve compose services like `mlflow:5000`
# through the outer `mlops` network without mounting the host Docker socket.

# TODO(you) #2 — not idempotent.
# Re-running this command registers a *second* runner with the same name.
# Either (a) detect existing registration and skip, or (b) `gitlab-runner unregister --all-runners` first.
# Bonus: store the resulting runner token in a docker volume so re-registration after a wipe is trivial. ✅

docker compose exec -T gitlab-runner gitlab-runner unregister --all-runners
docker compose exec -T gitlab-runner \
  gitlab-runner register \
    --non-interactive \
    --url "$GITLAB_URL" \
    --token "$GITLAB_RUNNER_TOKEN" \
    --executor docker \
    --docker-image "gcr.io/kaniko-project/executor:debug" \
    --description "local-dind-runner" \
    --docker-network-mode "host" # will run in network namespace of the dind container


# TODO(you) #3 — verification.
# The command above can succeed even when the runner can't actually reach
# GitLab (wrong URL, firewall, etc.). After registration, poll
# `docker compose exec gitlab-runner gitlab-runner verify` and assert all
# runners are alive. Fail the script if any are not. ✅
docker compose exec -T gitlab-runner gitlab-runner verify


echo "Runner registered. Verify in GitLab Admin Area -> Runners."

# TODO(you) #4 (stretch) — security. ✅
# The runner container no longer mounts /var/run/docker.sock; it talks to
# `gitlab-runner-dind` over TCP+TLS, so a compromised job cannot reach the
# host daemon. The sibling daemon is still privileged (any nested dockerd is)
# but its blast radius is one container + its volumes.
# Image builds use Kaniko, not docker:dind — see ml-app/.gitlab-ci.yml.
