#!/usr/bin/env bash
# Two-hosts lab: h1 <-> h2 via a veth pair.
# Run inside a Linux VM. See SETUP.md.
#
# Topology:
#
#   [h1: 10.0.0.1/24] --veth-- [h2: 10.0.0.2/24]
#
# After running, try:
#   sudo ip netns exec h1 ping -c 3 10.0.0.2
#   sudo ip netns exec h1 tcpdump -i veth-h1 icmp

set -euo pipefail

if [[ "$(id -u)" -ne 0 ]]; then
  echo "run with sudo" >&2
  exit 1
fi

ip netns add h1
ip netns add h2

ip link add veth-h1 type veth peer name veth-h2

ip link set veth-h1 netns h1
ip link set veth-h2 netns h2

ip -n h1 addr add 10.0.0.1/24 dev veth-h1
ip -n h2 addr add 10.0.0.2/24 dev veth-h2

ip -n h1 link set veth-h1 up
ip -n h2 link set veth-h2 up
ip -n h1 link set lo up
ip -n h2 link set lo up

cat <<EOF
ready.

  sudo ip netns exec h1 ip a
  sudo ip netns exec h1 ping -c 3 10.0.0.2
  sudo ip netns exec h1 tcpdump -ni veth-h1 -e

teardown:
  sudo bash teardown.sh
EOF
