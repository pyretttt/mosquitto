#!/usr/bin/env bash
set -euo pipefail

readonly SCENARIO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPTS_DIR="$(cd -- "${SCENARIO_DIR}/.." && pwd)"
source "${SCRIPTS_DIR}/common.sh"

main() {
    require_root
    "${SCRIPTS_DIR}/setup-routed.sh"
    "${SCRIPTS_DIR}/setup-services.sh"

    in_namespace "$CLP_SERVER" ip route delete default

    printf 'Scenario ready: routed path has a return-path fault.\n'
}

main "$@"
