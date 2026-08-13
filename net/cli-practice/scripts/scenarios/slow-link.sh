#!/usr/bin/env bash
set -euo pipefail

readonly SCENARIO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPTS_DIR="$(cd -- "${SCENARIO_DIR}/.." && pwd)"
source "${SCRIPTS_DIR}/common.sh"

main() {
    require_root
    "${SCRIPTS_DIR}/setup-routed.sh"
    "${SCRIPTS_DIR}/setup-services.sh"

    in_namespace "$CLP_ROUTER" tc qdisc add dev r-right root netem \
        delay 120ms loss 3% rate 5mbit

    printf 'Scenario ready: router-to-server traffic is impaired.\n'
}

main "$@"
