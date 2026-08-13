#!/usr/bin/env bash
set -euo pipefail

readonly CHECK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPTS_DIR="$(cd -- "${CHECK_DIR}/../scripts" && pwd)"
readonly CAPTURE_FILE="/tmp/clp-nat-check.capture"
source "${SCRIPTS_DIR}/common.sh"

capture_pid=""

stop_capture() {
    if [[ -n "$capture_pid" ]]; then
        kill "$capture_pid" >/dev/null 2>&1 || true
        wait "$capture_pid" >/dev/null 2>&1 || true
    fi
    rm -f "$CAPTURE_FILE"
}

observed_external_source() {
    local capture
    capture="$(<"$CAPTURE_FILE")"
    [[ "$capture" == *"198.51.100.254."* ]]
}

nat_is_working() {
    rm -f "$CAPTURE_FILE"
    timeout 5 ip netns exec "$CLP_SERVER" \
        tcpdump -c 1 -nn -i s-eth0 'tcp dst port 8080' \
        >"$CAPTURE_FILE" 2>/dev/null &
    capture_pid="$!"
    sleep 0.3

    in_namespace "$CLP_CLIENT" \
        curl --fail --silent --max-time 3 http://198.51.100.10:8080/ >/dev/null 2>&1 || return 1
    wait "$capture_pid" >/dev/null 2>&1 || return 1
    capture_pid=""
    observed_external_source
}

main() {
    require_root
    trap stop_capture EXIT

    if require_topology && nat_is_working; then
        printf 'PASS: translated service path is healthy.\n'
        return
    fi

    printf 'FAIL: translated service path is not healthy.\n'
    return 1
}

main "$@"
