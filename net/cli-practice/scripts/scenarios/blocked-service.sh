#!/usr/bin/env bash
set -euo pipefail

readonly SCENARIO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPTS_DIR="$(cd -- "${SCENARIO_DIR}/.." && pwd)"
source "${SCRIPTS_DIR}/common.sh"

main() {
    require_root
    "${SCRIPTS_DIR}/setup-routed.sh"
    "${SCRIPTS_DIR}/setup-services.sh"

    in_namespace "$CLP_ROUTER" nft add table inet clp_filter
    in_namespace "$CLP_ROUTER" nft \
        'add chain inet clp_filter forward { type filter hook forward priority filter; policy accept; }'
    in_namespace "$CLP_ROUTER" nft add rule inet clp_filter forward tcp dport 8080 drop

    printf 'Scenario ready: one forwarded service is blocked while ICMP remains available.\n'
}

main "$@"
