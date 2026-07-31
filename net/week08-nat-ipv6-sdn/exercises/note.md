## Learning goals

- Walk through SNAT/DNAT/MASQUERADE on Linux and what conntrack stores.
- Explain why two hosts behind the same NAT often *can't* reach each other via the NAT's public IP (hairpin NAT).
- Read an IPv6 address: link-local (fe80::/10), ULA (fc00::/7), GUA (2000::/3). Understand SLAAC, RA, DAD.
- Sketch what an OpenFlow "match-action" table looks like and why Kubernetes CNIs (esp. Cilium) like the model.

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
