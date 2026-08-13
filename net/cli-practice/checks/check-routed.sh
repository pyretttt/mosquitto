#!/usr/bin/env bash
set -euo pipefail

readonly CHECK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPTS_DIR="$(cd -- "${CHECK_DIR}/../scripts" && pwd)"
source "${SCRIPTS_DIR}/common.sh"

main() {
    require_root

    if require_topology && in_namespace "$CLP_CLIENT" ping -c 2 -W 1 10.20.2.10 >/dev/null 2>&1; then
        printf 'PASS: routed connectivity is healthy.\n'
        return
    fi

    printf 'FAIL: routed connectivity is not healthy.\n'
    return 1
}

main "$@"
