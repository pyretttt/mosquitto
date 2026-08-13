#!/usr/bin/env bash
set -euo pipefail

readonly CHECK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPTS_DIR="$(cd -- "${CHECK_DIR}/../scripts" && pwd)"
source "${SCRIPTS_DIR}/common.sh"

service_is_reachable() {
    in_namespace "$CLP_CLIENT" \
        curl --fail --silent --max-time 3 http://10.20.2.10:8080/ >/dev/null 2>&1 \
        && in_namespace "$CLP_CLIENT" nc -z -w 2 10.20.2.10 5201 >/dev/null 2>&1
}

main() {
    require_root

    if require_topology && service_is_reachable; then
        printf 'PASS: lab services are reachable.\n'
        return
    fi

    printf 'FAIL: lab services are not reachable.\n'
    return 1
}

main "$@"
