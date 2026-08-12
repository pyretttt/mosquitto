## Learning goals

- Walk through SNAT/DNAT/MASQUERADE on Linux and what conntrack stores.

SNAT _(source NAT)_  stores NAT entries for egress traffic. DNAT _(destination NAT)_ stores NAT entries for ingress traffic.

MSQUERADE is basically a SNAT with per flow lookup. The key distinction is that SNAT requires to specify address of translation. While MASQURADE look ups current address of output interface.

- Explain why two hosts behind the same NAT often *can't* reach each other via the NAT's public IP (hairpin NAT).

Consider flow
H2 (10.0.0.2) -> R (203.0.113.5:80)
Router NATs destination address to send packet to H3 (10.0.0.3)
H3 accepts traffic from (10.0.0.2) understand that peer is inside local network and sends response with (src=10.0.0.2)
H2 declines response because it waits for traffic from (203.0.113.5:80)

To fix it router must also change `source port`


- Read an IPv6 address: link-local (fe80::/10), ULA (fc00::/7), GUA (2000::/3). Understand SLAAC, RA, DAD.

link-local (fe80::/10) is link unique address that is assigned by peer itself. It can be derived with EUI-64 or stable stable-privacy addresses.

Unique local addresses and Global Unique addresses are usually derived from the same IID that is used to form final IP address.

- Sketch what an OpenFlow "match-action" table looks like and why Kubernetes CNIs (esp. Cilium) like the model.

```
priority  match                                            actions
--------  -----------------------------------------------  --------------------
   200    in_port=1, ip, nw_dst=10.0.0.5                    output:3
   200    in_port=1, ip, nw_dst=10.0.0.6                    output:4
   100    in_port=1, arp                                    flood
   100    dl_dst=00:11:22:33:44:55                          output:2
    50    ip, nw_dst=10.0.0.0/24                            goto_table:2
     0    *                                                 drop <- table-miss
```

## Exercises

1. **Conceptual.** In `notes.md`:
   - Why does NAT break end-to-end connectivity (the original IP architecture promise)?

The original IP architecture made a specific promise: **every host has a globally unique address, every host can send a packet to any other, and all per-connection state lives in endpoints**. Routers were supposed to be stateless forwarders that could be swapped out mid-conversation without anyone noticing. Nat violates all three parts. They fail in different ways


   - What's the difference between a "full cone", "restricted cone", and "symmetric" NAT?

**Full cone** reuses ports for the same origin port. Any inbound to source address are accepted. **Restricted cone** reuses ports for the same origin port. Any port from destination address can send to NATed address. **Symmetric NAT** do not reuse ports across different destination, even for the same port. Allowed only Inbound from destination address with the same port.

   - In IPv6, what's the role of `fe80::/10` and why does every interface always have one?

It's subnet of unique local addresses inside link. It's uniqueness guaranteed only over link. It's required because without link-local address we can't run DAD for ULA/GUA. We can't ask for Router Advertisement. Address resolution stops, every NS/NA exchange that turns an address into a MAC needs both parties addressable on the link

2. **Practical - hairpin NAT.** Reproduce the failure, then fix it with a `iptables`/`nft` rule. Save before/after captures.

Done in `nat-hairpin.sh`

3. **Practical - nftables vs iptables.** Re-implement the MASQUERADE setup in both (`nft`) and (`iptables`). Compare the rule listing output. Save in `exercises/08-nat-rules.md`.
4. **Stretch - SDN intro.** Install `ovs-vsctl` (Open vSwitch), connect two netns through OVS, and play with flow rules:
   ```bash
   ovs-ofctl add-flow br0 "in_port=1,actions=output:2"
   ovs-ofctl dump-flows br0
   ```

## Self-check

- Without notes: what does a `MASQUERADE` rule actually do at packet time, and what does `conntrack` keep so the return packet is rewritten correctly?

1. First routing decision is made, i.e. output device is selected.
2. MASQUERADE reads primary IPv4 address of this device
3. Performs SNAT to that address, allocation a source port
4. Records the resulting binding on the conntrack entry

- What's an EUI-64 IPv6 address and why might you not want it in 2026?

EUI-64 IPv6 is IID made from device MAC address, it gives stable address for network device on host. It should not be used becaues it exposes device across multiple networks. As soon as same local link address used across networks.

- How does a managed AWS NAT Gateway differ from `iptables -j MASQUERADE` on a Linux box?

1. **The connection limit is per-destination, not global.** 55,000 sounds tiny until you realise it
   is 55,000 to *each* `(ip, port, protocol)`. It only bites when a fleet hammers a single endpoint
   — an S3 VPC endpoint bypass, one busy API, a shared database. That is also the exact shape of
   the Linux failure: `insert_failed` from port exhaustion for one destination tuple, not from the
   table being globally full. Same bug, two dashboards.
2. **Source IP selection is non-deterministic across a multi-IP gateway.** MASQUERADE gives you one
   predictable egress address per interface; NAT Gateway with several addresses hashes per flow,
   including the TCP sequence number in the hash. Any downstream partner doing source-IP
   allowlisting must allowlist all of them, and you cannot reason about which one a given instance
   will use.
3. **350 seconds.** This is the single most common production surprise. A long-lived idle
   connection — a database pool, a gRPC channel, a queue consumer — dies silently and the
   application discovers it on the next write. Set TCP keepalives below 350s. There is no
   equivalent trap on Linux with its five-day default, which is exactly why people are unprepared
   for it.

- In SDN parlance, what's the data plane vs control plane? Where does Cilium fit?

Data plane is per packet routing routines. Control plane is network wide routing routines, usually control plane is used to create data plane routing rules to later update middle boxes.

Cilium control plane runs on each node. It watches the API, computes desired states, loads eBPF programs and writes maps.

Data plane - the eBPF programs and maps in the kernel. It keeps working if agent dies. Cilium is not one central OpenFlow controller programming every switch.