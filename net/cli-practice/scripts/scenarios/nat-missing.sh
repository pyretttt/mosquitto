#!/usr/bin/env bash
set -euo pipefail

readonly SCENARIO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPTS_DIR="$(cd -- "${SCENARIO_DIR}/.." && pwd)"
source "${SCRIPTS_DIR}/common.sh"

cleanup_failed_setup() {
    "${SCRIPTS_DIR}/reset.sh" >/dev/null 2>&1 || true
    ip link delete clp-nin-tmp >/dev/null 2>&1 || true
    ip link delete clp-nout-tmp >/dev/null 2>&1 || true
}

create_link() {
    local first_namespace="$1"
    local first_interface="$2"
    local second_namespace="$3"
    local second_interface="$4"
    local temporary_interface="$5"
    local temporary_peer="$6"

    ip link add "$temporary_interface" type veth peer name "$temporary_peer"
    ip link set "$temporary_interface" netns "$first_namespace"
    ip link set "$temporary_peer" netns "$second_namespace"
    in_namespace "$first_namespace" ip link set "$temporary_interface" name "$first_interface"
    in_namespace "$second_namespace" ip link set "$temporary_peer" name "$second_interface"
}

configure_interface() {
    local namespace="$1"
    local interface="$2"
    local address="$3"

    in_namespace "$namespace" ip address add "$address" dev "$interface"
    in_namespace "$namespace" ip link set "$interface" up
}

main() {
    require_root
    trap cleanup_failed_setup ERR

    "${SCRIPTS_DIR}/reset.sh" >/dev/null
    create_namespace "$CLP_CLIENT"
    create_namespace "$CLP_ROUTER"
    create_namespace "$CLP_SERVER"

    create_link "$CLP_CLIENT" c-eth0 "$CLP_ROUTER" r-inside clp-nin-tmp clp-nin-peer
    create_link "$CLP_ROUTER" r-outside "$CLP_SERVER" s-eth0 clp-nout-tmp clp-nout-peer

    configure_interface "$CLP_CLIENT" c-eth0 10.30.1.10/24
    configure_interface "$CLP_ROUTER" r-inside 10.30.1.1/24
    configure_interface "$CLP_ROUTER" r-outside 198.51.100.254/24
    configure_interface "$CLP_SERVER" s-eth0 198.51.100.10/24

    in_namespace "$CLP_CLIENT" ip route add default via 10.30.1.1
    in_namespace "$CLP_ROUTER" sysctl -q -w net.ipv4.ip_forward=1

    trap - ERR
    "${SCRIPTS_DIR}/setup-services.sh"
    printf 'Scenario ready: private client path has no address translation.\n'
}

main "$@"
