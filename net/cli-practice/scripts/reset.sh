#!/usr/bin/env bash
set -euo pipefail

readonly RESET_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${RESET_DIR}/common.sh"

stop_namespace_processes() {
    local namespace="$1"
    local -a process_ids=()

    mapfile -t process_ids < <(ip netns pids "$namespace")
    if ((${#process_ids[@]} == 0)); then
        return
    fi

    kill "${process_ids[@]}" 2>/dev/null || true
    sleep 0.2

    mapfile -t process_ids < <(ip netns pids "$namespace")
    if ((${#process_ids[@]} > 0)); then
        kill -KILL "${process_ids[@]}" 2>/dev/null || true
    fi
}

delete_namespace() {
    local namespace="$1"
    [[ "$namespace" == clp-* ]] || {
        printf 'Refusing to delete namespace without clp- prefix.\n' >&2
        exit 1
    }

    namespace_exists "$namespace" || return 0
    stop_namespace_processes "$namespace"
    ip netns delete "$namespace"
}

main() {
    require_root

    local namespace
    for namespace in "${CLP_NAMESPACES[@]}"; do
        delete_namespace "$namespace"
    done

    rm -f /run/clp-http.pid /run/clp-iperf3.pid /tmp/clp-http.log /tmp/clp-iperf3.log
    printf 'Lab namespaces reset.\n'
}

main "$@"
