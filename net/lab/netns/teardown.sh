#!/usr/bin/env bash
# Best-effort teardown. Removes all netns we may have created.
# Idempotent - missing namespaces are silently ignored.

set -u

if [[ "$(id -u)" -ne 0 ]]; then
  echo "run with sudo" >&2
  exit 1
fi

for ns in h1 h2 h3 r r1 r2 r3 r4 public bridge sw1 sw2 client server; do
  ip netns delete "$ns" 2>/dev/null || true
done

for link in veth-h1 veth-h2 veth-rh1 veth-rh2 veth-rin veth-rout veth-pub; do
  ip link delete "$link" 2>/dev/null || true
done

echo "torn down (any remaining custom netns/links you should remove manually)"
ip netns list
