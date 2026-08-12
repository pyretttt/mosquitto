#!/usr/bin/env bash
# Three-node routed dual-stack lab: h1 <-> r <-> h2.
# r is a router with two NICs; IPv4 + IPv6 forwarding enabled.
# Link-locals are left to the kernel; we assign ULA for routed v6.
#
# Topology:
#
#   [h1: 10.0.0.1/24 , fd00:0:0:0::1/64]
#        |
#   [r:  10.0.0.254/24 , fd00:0:0:0::254/64
#        10.0.1.254/24 , fd00:0:0:1::254/64]
#        |
#   [h2: 10.0.1.1/24 , fd00:0:0:1::1/64]
#
# After running, try:
#   sudo ip netns exec h1 ping  -c 3 10.0.1.1
#   sudo ip netns exec h1 ping6 -c 3 fd00:0:0:1::1
#   sudo ip netns exec h1 traceroute -n 10.0.1.1
#   sudo ip netns exec r tcpdump -ni any -e

set -euo pipefail

if [[ "$(id -u)" -ne 0 ]]; then
  echo "run with sudo" >&2
  exit 1
fi

for ns in h1 r h2; do
  ip netns add "$ns"
done

ip link add veth-h1 type veth peer name veth-rh1
ip link add veth-h2 type veth peer name veth-rh2

ip link set veth-h1  netns h1
ip link set veth-rh1 netns r
ip link set veth-rh2 netns r
ip link set veth-h2  netns h2

ip -n h1 addr add 10.0.0.1/24        dev veth-h1
ip -n h1 addr add fd00:0:0:0::1/64   dev veth-h1
ip -n r  addr add 10.0.0.254/24      dev veth-rh1
ip -n r  addr add fd00:0:0:0::254/64 dev veth-rh1
ip -n r  addr add 10.0.1.254/24      dev veth-rh2
ip -n r  addr add fd00:0:0:1::254/64 dev veth-rh2
ip -n h2 addr add 10.0.1.1/24        dev veth-h2
ip -n h2 addr add fd00:0:0:1::1/64   dev veth-h2

for ns in h1 r h2; do ip -n "$ns" link set lo up; done
ip -n h1 link set veth-h1 up
ip -n r  link set veth-rh1 up
ip -n r  link set veth-rh2 up
ip -n h2 link set veth-h2 up

ip -n h1 route add default via 10.0.0.254
ip -n h2 route add default via 10.0.1.254
ip -n h1 -6 route add default via fd00:0:0:0::254
ip -n h2 -6 route add default via fd00:0:0:1::254

ip netns exec r sysctl -w net.ipv4.ip_forward=1 >/dev/null
ip netns exec r sysctl -w net.ipv6.conf.all.forwarding=1 >/dev/null

cat <<EOF
ready.

  sudo ip netns exec h1 ping  -c 3 10.0.1.1
  sudo ip netns exec h1 ping6 -c 3 fd00:0:0:1::1
  sudo ip netns exec h1 traceroute -n 10.0.1.1
  sudo ip netns exec h1 traceroute -6 -n fd00:0:0:1::1
  sudo ip netns exec r  ip -6 route
  sudo ip netns exec r  tcpdump -ni any -e icmp or icmp6

useful break-it experiments:
  # disable v4 or v6 forwarding independently
  sudo ip netns exec r sysctl -w net.ipv4.ip_forward=0
  sudo ip netns exec r sysctl -w net.ipv6.conf.all.forwarding=0
  # remove h2's default route -> ping works one way
  sudo ip netns exec h2 ip route del default
  sudo ip netns exec h2 ip -6 route del default
  # set MTU low on both ends of the r<->h2 link, then big ping
  sudo ip netns exec r  ip link set veth-rh2 mtu 800
  sudo ip netns exec h2 ip link set veth-h2  mtu 800
  sudo ip netns exec h1 ping -c 1 -s 1400 -M dont 10.0.1.1   # OK, fragments
  sudo ip netns exec h1 ping -c 1 -s 1400 -M do   10.0.1.1   # Frag needed

teardown:
  sudo bash teardown.sh
EOF
