#!/usr/bin/env bash
# NAT gateway lab: simulate a "private host behind a NAT" scenario.
#
# Topology:
#
#   [h1: 10.0.0.1/24]  ----  [r: 10.0.0.254 / 198.51.100.254]  ----  [public: 198.51.100.1/24]
#                              ^ does SNAT/MASQUERADE
#
# After running, try:
#   sudo ip netns exec h1 ping -c 3 198.51.100.1
#   sudo ip netns exec r conntrack -L
#
# Note that 198.51.100.0/24 is TEST-NET-2 (RFC 5737). It is reserved for documentation
# but works fine on virtual interfaces - we just pretend it is "the internet".

set -euo pipefail

if [[ "$(id -u)" -ne 0 ]]; then
  echo "run with sudo" >&2
  exit 1
fi

for ns in h1 r public; do
  ip netns add "$ns"
done

ip link add veth-h1 type veth peer name veth-rin
ip link add veth-rout type veth peer name veth-pub

ip link set veth-h1   netns h1
ip link set veth-rin  netns r
ip link set veth-rout netns r
ip link set veth-pub  netns public

ip -n h1     addr add 10.0.0.1/24       dev veth-h1
ip -n r      addr add 10.0.0.254/24     dev veth-rin
ip -n r      addr add 198.51.100.254/24 dev veth-rout
ip -n public addr add 198.51.100.1/24   dev veth-pub

for ns in h1 r public; do ip -n "$ns" link set lo up; done
ip -n h1     link set veth-h1   up
ip -n r      link set veth-rin  up
ip -n r      link set veth-rout up
ip -n public link set veth-pub  up

ip -n h1     route add default via 10.0.0.254
ip -n public route add default via 198.51.100.254

ip netns exec r sysctl -w net.ipv4.ip_forward=1 >/dev/null

ip netns exec r nft -f - <<'NFT'
table inet nat {
    chain postrouting {
        type nat hook postrouting priority 100;
        oifname "veth-rout" masquerade
    }
}
NFT

cat <<EOF
ready.

  sudo ip netns exec h1     ping -c 3 198.51.100.1
  sudo ip netns exec r      conntrack -L
  sudo ip netns exec public tcpdump -ni veth-pub -e

  # Run nc on public, connect from h1, watch source IP rewriting:
  sudo ip netns exec public nc -lvnp 8080 &
  sudo ip netns exec h1     bash -c 'echo hello | nc 198.51.100.1 8080'

teardown:
  sudo bash teardown.sh
EOF
