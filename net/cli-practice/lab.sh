#!/usr/bin/env bash
set -euo pipefail

readonly LAB_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly COMPOSE_FILE="${LAB_DIR}/compose.yaml"

usage() {
    printf 'Usage: %s {up|shell|down|rebuild}\n' "$(basename -- "$0")" >&2
}

require_docker() {
    command -v docker >/dev/null 2>&1 || {
        printf 'Docker is required.\n' >&2
        exit 1
    }
    docker compose version >/dev/null 2>&1 || {
        printf 'Docker Compose is required.\n' >&2
        exit 1
    }
}

compose() {
    docker compose --project-directory "$LAB_DIR" --file "$COMPOSE_FILE" "$@"
}

main() {
    require_docker

    case "${1:-}" in
        up)
            compose up --detach
            ;;
        shell)
            compose up --detach
            compose exec lab bash
            ;;
        down)
            compose down --remove-orphans
            ;;
        rebuild)
            compose build --no-cache
            compose up --detach --force-recreate
            ;;
        *)
            usage
            exit 2
            ;;
    esac
}

main "$@"
