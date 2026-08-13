#!/usr/bin/env bash
set -euo pipefail

readonly CLP_CLIENT="clp-client"
readonly CLP_ROUTER="clp-router"
readonly CLP_SERVER="clp-server"
readonly CLP_NAMESPACES=("$CLP_CLIENT" "$CLP_ROUTER" "$CLP_SERVER")
readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

require_root() {
    if [[ "${EUID}" -ne 0 ]]; then
        printf 'Run this script as root inside the lab container.\n' >&2
        exit 1
    fi
}

namespace_exists() {
    local namespace="$1"
    ip netns list | awk '{print $1}' | awk -v expected="$namespace" '$0 == expected { found=1 } END { exit !found }'
}

require_topology() {
    local namespace
    for namespace in "${CLP_NAMESPACES[@]}"; do
        namespace_exists "$namespace" || {
            printf 'Required lab topology is not running.\n' >&2
            return 1
        }
    done
}

in_namespace() {
    local namespace="$1"
    shift
    ip netns exec "$namespace" "$@"
}

create_namespace() {
    local namespace="$1"
    [[ "$namespace" == clp-* ]] || {
        printf 'Refusing to create namespace without clp- prefix.\n' >&2
        exit 1
    }
    ip netns add "$namespace"
    in_namespace "$namespace" ip link set lo up
}
