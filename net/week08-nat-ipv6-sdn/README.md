# Week 8 - Ch4 part 2: NAT, IPv6, SDN data plane

NAT is the glue that holds IPv4 together. IPv6 is the alternate universe where NAT mostly doesn't exist. SDN data-plane is the abstraction Kubernetes networking is built on.

## Reading

- Kurose & Ross 8e, 4.4-4.5 (~30 pages):
  - 4.4 Generalized forwarding and SDN
  - 4.5 Middleboxes (NAT, firewalls)
- RFC 8200 (IPv6 base)
- RFC 4862 (IPv6 SLAAC)
- RFC 4787 (NAT behavioral requirements for UDP)
- (optional) RFC 5128 ("State of P2P communication across NATs") for hairpin / cone-NAT terminology.
- (optional) "OpenFlow: enabling innovation in campus networks" (McKeown et al., 2008).

## Learning goals

- Walk through SNAT/DNAT/MASQUERADE on Linux and what conntrack stores.
- Explain why two hosts behind the same NAT often *can't* reach each other via the NAT's public IP (hairpin NAT).
- Read an IPv6 address: link-local (fe80::/10), ULA (fc00::/7), GUA (2000::/3). Understand SLAAC, RA, DAD.
- Sketch what an OpenFlow "match-action" table looks like and why Kubernetes CNIs (esp. Cilium) like the model.

## Lab

**Goal:** stand up a NAT gateway in netns, watch conntrack, then reproduce hairpin NAT.

1. Bring up the NAT lab:
   ```bash
   cd lab/netns
   sudo bash nat-gateway.sh
   ```
2. Generate traffic and watch the conntrack table:
   ```bash
   sudo ip netns exec public bash -c 'while true; do nc -l -p 8080; done' &
   sudo ip netns exec h1     bash -c 'echo hello | nc 198.51.100.1 8080'
   sudo ip netns exec r      conntrack -L
   ```
3. Capture from "the public side" to see only the post-NAT addresses:
   ```bash
   sudo ip netns exec public tcpdump -ni veth-pub -e
   ```
4. **Hairpin NAT.** Add a second host h3 behind the same NAT (extend the script). Try to make h3 talk to a service on h1 via the NAT's public IP `198.51.100.254`. It will fail unless you add a hairpin rule. Document why.
5. **IPv6 dual-stack.** Add `fd00:1::/64` link-local + a ULA (`fd00:abcd::/48`) on the topology, configure SLAAC (`radvd` on r), watch RA messages with `tcpdump icmp6`.

## Exercises

1. **Conceptual.** In `notes.md`:
   - Why does NAT break end-to-end connectivity (the original IP architecture promise)?
   - What's the difference between a "full cone", "restricted cone", and "symmetric" NAT?
   - In IPv6, what's the role of `fe80::/10` and why does every interface always have one?
2. **Practical - hairpin NAT.** Reproduce the failure, then fix it with a `iptables`/`nft` rule. Save before/after captures.
3. **Practical - nftables vs iptables.** Re-implement the MASQUERADE setup in both (`nft`) and (`iptables`). Compare the rule listing output. Save in `exercises/08-nat-rules.md`.
4. **Stretch - SDN intro.** Install `ovs-vsctl` (Open vSwitch), connect two netns through OVS, and play with flow rules:
   ```bash
   ovs-ofctl add-flow br0 "in_port=1,actions=output:2"
   ovs-ofctl dump-flows br0
   ```

## Self-check

- Without notes: what does a `MASQUERADE` rule actually do at packet time, and what does `conntrack` keep so the return packet is rewritten correctly?
- What's an EUI-64 IPv6 address and why might you not want it in 2026?
- How does a managed AWS NAT Gateway differ from `iptables -j MASQUERADE` on a Linux box?
- In SDN parlance, what's the data plane vs control plane? Where does Cilium fit?

## Useful one-liners

```bash
conntrack -L                              # current connections
conntrack -E                              # event stream (great for debugging)
sysctl net.netfilter.nf_conntrack_count
sysctl net.netfilter.nf_conntrack_max

ip -6 addr
ip -6 route
ip -6 neigh                               # IPv6 ARP equivalent (NDP)
```
