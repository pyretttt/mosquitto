# Week 7 - Ch4 part 1: Forwarding, IP, fragmentation

You drop down a layer. This week is "what does a router actually do, and what's in an IP header." The PMTUD black-hole exercise is the one that makes everything click.

## Reading

- Kurose & Ross 8e, 4.1-4.3 (~50 pages):
  - 4.1 Overview of network layer
  - 4.2 What's inside a router
  - 4.3 The Internet Protocol (IPv4): forwarding, addressing, fragmentation
- RFC 791 (IPv4) - read sections 1-3.
- RFC 1191 (Path MTU Discovery)
- (skim) RFC 4459 ("MTU and Fragmentation Issues with In-the-Network Tunneling")

## Learning goals

- Explain longest-prefix match and the *forwarding table* vs *routing table* distinction.
- Read every IPv4 header field: version, IHL, DSCP/ECN, total length, ID, flags (DF/MF), fragment offset, TTL, protocol, checksum, src, dst.
- Reason about MTU vs MSS, why DF=1 + ICMP-blocked router = PMTUD black hole.
- Fix a black hole with TCP MSS clamping.

## Lab

**Goal:** make a routed topology, observe fragmentation, and reproduce + fix a PMTUD black hole.

1. Bring up the router topology:
   ```bash
   cd lab/netns
   sudo bash router-3-nodes.sh
   ```
2. Confirm forwarding works:
   ```bash
   sudo ip netns exec h1 ping -c 3 10.0.1.1
   sudo ip netns exec h1 traceroute -n 10.0.1.1
   ```
3. Force fragmentation. Lower the MTU on r's outbound:
   ```bash
   sudo ip netns exec r ip link set veth-rh2 mtu 800
   sudo ip netns exec h1 ping -c 1 -s 1400 10.0.1.1            # OK, fragments
   sudo ip netns exec h1 ping -c 1 -s 1400 -M do 10.0.1.1      # fails: needs frag, DF set
   ```
4. Capture and inspect:
   ```bash
   sudo ip netns exec r tcpdump -ni any -w /tmp/frag.pcap -e icmp or 'ip[6] & 0x1f != 0'
   ```
5. Reproduce the **PMTUD black hole**: drop ICMP "Frag Needed" on the router:
   ```bash
   sudo ip netns exec r nft add rule inet filter forward icmp type destination-unreachable drop
   sudo ip netns exec h1 curl -s -o /dev/null --max-time 5 http://10.0.1.1   # if there's a server, will hang
   ```
   The TCP handshake completes (small SYN packets) but the first big response packet from the server gets silently dropped; PMTUD never works because the ICMP is filtered.
6. **Fix it** with MSS clamping:
   ```bash
   sudo ip netns exec r nft add rule inet filter forward tcp flags syn tcp option maxseg size set rt mtu
   ```
   (the equivalent of `iptables -t mangle ... --clamp-mss-to-pmtu`)

## Exercises

1. **Conceptual.** In `notes.md`:
   - Sketch the IPv4 header field by field. For each, say what changes hop-to-hop (TTL, checksum) vs end-to-end (src, dst, ID).
   - Why doesn't IPv6 do in-flight fragmentation? Where did that responsibility move to?
   - Solve 2-3 end-of-chapter problems on subnetting / longest prefix match.
2. **Practical.** Reproduce the PMTUD black hole + MSS clamping fix end-to-end. Save the pcaps before/after as `captures/07-pmtud-black-hole.pcap` and `captures/07-pmtud-clamped.pcap`. Write a 1-page postmortem in `exercises/07-postmortem.md` as if it had hit production.
3. **Stretch.** Set up a GRE tunnel between h1 and h2 over r and watch the encapsulated MTU shrink to 1476. Send a 1500-byte ping with DF and see exactly what breaks.

## Self-check

- Without notes: what's *longest prefix match* and why is it the rule?
- Why does the IPv4 header have a checksum but the IPv6 header doesn't?
- What's the difference between MTU and MSS? How are they negotiated?
- What's a PMTUD black hole? How do you detect one?
- What's TCP MSS clamping and where is it usually configured?

## Useful one-liners

```bash
ip route get 10.0.1.1                    # which route would match?
ip route show table all
ping -M do -s 1500 10.0.1.1              # DF=1, no frag
tracepath 10.0.1.1                       # discovers PMTU as it goes
```
