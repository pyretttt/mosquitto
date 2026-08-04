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

for ns in h1 h3 r public; do
  ip netns add "$ns"
done

ip link add veth-h1 type veth peer name veth1-rin
ip link add veth-h3 type veth peer name veth3-rin
ip link add veth-rout type veth peer name veth-pub

ip link set veth-h1   netns h1
ip link set veth-h3   netns h3
ip link set veth1-rin  netns r
ip link set veth3-rin  netns r
ip link set veth-rout netns r
ip link set veth-pub  netns public

ip -n r link add br-in type bridge
ip -n r link set veth1-rin master br-in
ip -n r link set veth3-rin master br-in

ip -n h1     addr add 10.0.0.1/24       dev veth-h1
ip -n h3     addr add 10.0.0.3/24       dev veth-h3
ip -n r      addr add 198.51.100.254/24 dev veth-rout
ip -n public addr add 198.51.100.1/24   dev veth-pub
ip -n r addr add 10.0.0.254/24 dev br-in

for ns in h1 h3 r public; do ip -n "$ns" link set lo up; done
ip -n h1     link set veth-h1   up
ip -n h3     link set veth-h3   up
ip -n r      link set veth1-rin  up
ip -n r      link set veth3-rin  up
ip -n r      link set veth-rout up
ip -n r      link set br-in up
ip -n public link set veth-pub  up

ip -n h1     route add default via 10.0.0.254
ip -n h3     route add default via 10.0.0.254
ip -n public route add default via 198.51.100.254

ip netns exec r sysctl -w net.ipv4.ip_forward=1 >/dev/null

# DNAT + MASQUERADE on the router. DNAT is intentionally unscoped (no iifname)
# so hairpin from h3 also matches — that is the bug this lab reproduces.
# This works because h1 -> h3 goes not directly but by the br-in bridge.
ip netns exec r nft -f - <<'NFT'
table inet nat {
    chain prerouting {
        type nat hook prerouting priority dstnat;
        tcp dport 8080 dnat ip to 10.0.0.1:8080
    }
    chain postrouting {
        type nat hook postrouting priority srcnat;
        oifname "veth-rout" masquerade
    }
}
NFT

cat <<EOF
ready.

  # 1) Service on h1 (keep this running):
  sudo ip netns exec h1 bash -c 'while true; do nc -l -p 8080; done'

  # 2) Works from outside:
  sudo ip netns exec public bash -c 'echo hello | nc -v 198.51.100.254 8080'

  # 3) Fails from inside (hairpin) — hang/RST:
  sudo ip netns exec h3 bash -c 'echo hello | nc -v 198.51.100.254 8080'

  # Diagnostics while (3) is failing:
  sudo ip netns exec h3 tcpdump -ni veth-h3 -e
  sudo ip netns exec r  conntrack -E
  sudo ip netns exec r  nft list ruleset

  # Later fix (hairpin SNAT) — do NOT add until you have seen the failure:
  # sudo ip netns exec r nft add rule inet nat postrouting \
  #   ip saddr 10.0.0.0/24 ip daddr 10.0.0.1 tcp dport 8080 snat to 10.0.0.254

teardown:
  sudo bash teardown.sh
EOF
