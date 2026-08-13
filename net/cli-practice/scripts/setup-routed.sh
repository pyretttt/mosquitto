#!/usr/bin/env bash
set -euo pipefail

readonly SETUP_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SETUP_DIR}/common.sh"

cleanup_failed_setup() {
    "${SETUP_DIR}/reset.sh" >/dev/null 2>&1 || true
    ip link delete clp-left-tmp >/dev/null 2>&1 || true
    ip link delete clp-right-tmp >/dev/null 2>&1 || true
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

    "${SETUP_DIR}/reset.sh" >/dev/null
    create_namespace "$CLP_CLIENT"
    create_namespace "$CLP_ROUTER"
    create_namespace "$CLP_SERVER"

    create_link "$CLP_CLIENT" c-eth0 "$CLP_ROUTER" r-left clp-left-tmp clp-left-peer
    create_link "$CLP_ROUTER" r-right "$CLP_SERVER" s-eth0 clp-right-tmp clp-right-peer

    configure_interface "$CLP_CLIENT" c-eth0 10.20.1.10/24
    configure_interface "$CLP_ROUTER" r-left 10.20.1.1/24
    configure_interface "$CLP_ROUTER" r-right 10.20.2.1/24
    configure_interface "$CLP_SERVER" s-eth0 10.20.2.10/24

    in_namespace "$CLP_CLIENT" ip route add default via 10.20.1.1
    in_namespace "$CLP_SERVER" ip route add default via 10.20.2.1
    in_namespace "$CLP_ROUTER" sysctl -q -w net.ipv4.ip_forward=1

    trap - ERR
    printf '%s\n' \
        'clp-client 10.20.1.10/24 (c-eth0)' \
        '    -> clp-router 10.20.1.1/24 (r-left)' \
        '    -> clp-router 10.20.2.1/24 (r-right)' \
        '    -> clp-server 10.20.2.10/24 (s-eth0)'
}

main "$@"
