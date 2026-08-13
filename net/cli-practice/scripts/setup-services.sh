#!/usr/bin/env bash
set -euo pipefail

readonly SERVICES_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SERVICES_DIR}/common.sh"

http_ready() {
    in_namespace "$CLP_SERVER" curl --fail --silent --max-time 1 http://127.0.0.1:8080/ >/dev/null
}

iperf_ready() {
    in_namespace "$CLP_SERVER" nc -z -w 1 127.0.0.1 5201 >/dev/null 2>&1
}

wait_until_ready() {
    local description="$1"
    local readiness_function="$2"
    local attempt

    for attempt in {1..30}; do
        if "$readiness_function"; then
            return
        fi
        sleep 0.1
    done

    printf '%s failed to become ready.\n' "$description" >&2
    exit 1
}

start_http() {
    http_ready && return
    in_namespace "$CLP_SERVER" \
        python3 -u -m http.server 8080 --bind 0.0.0.0 \
        >/tmp/clp-http.log 2>&1 &
    printf '%s\n' "$!" >/run/clp-http.pid
}

start_iperf() {
    iperf_ready && return
    in_namespace "$CLP_SERVER" iperf3 --server \
        >/tmp/clp-iperf3.log 2>&1 &
    printf '%s\n' "$!" >/run/clp-iperf3.pid
}

main() {
    require_root
    require_topology

    start_http
    start_iperf
    wait_until_ready 'HTTP service' http_ready
    wait_until_ready 'iperf3 service' iperf_ready

    printf 'Services ready on clp-server: TCP/8080 and TCP/5201.\n'
}

main "$@"
