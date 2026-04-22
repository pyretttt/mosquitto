#!/usr/bin/env bash
# Register the GitLab Runner container against the local GitLab instance.
#
# Prereqs:
#   1. `docker compose up -d gitlab` and wait until http://localhost:8080 loads.
#   2. Reset the root password (on first boot):
#         docker compose exec gitlab grep 'Password:' /etc/gitlab/initial_root_password
#      (that file is auto-deleted after 24h; or just run
#       `docker compose exec -it gitlab gitlab-rake "gitlab:password:reset"`).
#   3. In the GitLab UI: Admin Area -> CI/CD -> Runners -> New instance runner.
#      Copy the *registration token*.
#
# Then run:  ./scripts/register-runner.sh <REGISTRATION_TOKEN>
#
# The runner will be registered with a `docker` executor that pulls images from
# Docker Hub and has Docker-in-Docker enabled (so the pipeline can build
# images). This is fine for local learning; not safe for real infra.

set -euo pipefail

TOKEN="${1:-}"
if [[ -z "$TOKEN" ]]; then
  echo "usage: $0 <REGISTRATION_TOKEN>" >&2
  exit 2
fi

# Inside the compose network, GitLab is reachable as http://gitlab/.
# From the runner's POV that hostname resolves via Docker's DNS.
GITLAB_URL="${GITLAB_URL:-http://gitlab}"

docker compose exec -T gitlab-runner \
  gitlab-runner register \
    --non-interactive \
    --url "$GITLAB_URL" \
    --registration-token "$TOKEN" \
    --executor docker \
    --docker-image "docker:27" \
    --docker-privileged \
    --docker-volumes "/var/run/docker.sock:/var/run/docker.sock" \
    --docker-network-mode "mlops_mlops" \
    --description "local-docker-runner" \
    --tag-list "docker,local" \
    --run-untagged="true" \
    --locked="false"

echo "Runner registered. Verify in GitLab Admin Area -> Runners."
