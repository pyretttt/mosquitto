## Exercises

1. **Conceptual.** In `notes.md`:
   - Sketch the IPv4 header field by field. For each, say what changes hop-to-hop (TTL, checksum) vs end-to-end (src, dst, ID).

```
| version | header length | TOS | Datagram length (bytes)      |
| 16 bit identifier             | flags | Fragmentation offset |
| TTL             | Upper Proto | Header checksum              |
| Source IP address                                            |
| Destination IP address                                       |
| Options if any                                               |
| payload                                                      |
```

   - Why doesn't IPv6 do in-flight fragmentation? Where did that responsibility move to?

Primary concern is performance in-between nodes do not do fragmentation. It's goal of end systems now. Datagram structure also moved fragmentation fields up to payload.

   - Solve 2-3 end-of-chapter problems on subnetting / longest prefix match.

Done

2. **Practical.** Reproduce the PMTUD black hole + MSS clamping fix end-to-end. Save the pcaps before/after as `captures/07-pmtud-black-hole.pcap` and `captures/07-pmtud-clamped.pcap`. Write a 1-page postmortem in `exercises/07-postmortem.md` as if it had hit production.
3. **Stretch.** Set up a GRE tunnel between h1 and h2 over r and watch the encapsulated MTU shrink to 1476. Send a 1500-byte ping with DF and see exactly what breaks.

## Self-check

- Without notes: what's *longest prefix match* and why is it the rule?

Router routing table contains prefixes that define where should datagram be forwarded. Rule states that if multiple prefixes is matched, the longest prefix is selected.

- Why does the IPv4 header have a checksum but the IPv6 header doesn't?

IPv4 contains checksum, because IP may be used with protocols that don't integrity checking. But IPv6 get rid of it, mostly because of performance considerations (checksum must be recomputed each time ttl decremented)

- What's the difference between MTU and MSS? How are they negotiated?

MTU is maximum transfer unit (Network layer IP) by default it's 1500 bytes, but can be changed at interface level
MSS is maximum segment size (Transport layer) - negotiated through options inside tcp segment. Usually it is equal to `MTU - ip header - tcp header`

- What's a PMTUD black hole? How do you detect one?

A PMTU black hole happens when Path MTU Discovery fails silently — a router along the path needs to fragment an oversized packet, but can't send the ICMP message telling the sender to shrink its packets, and the packet just disappears with no error returned. The sender never learns it needs to reduce its packet size, so it keeps retransmitting the same oversized packet, which keeps disappearing. Connection just hangs.

Hot to detect it

Ping with DF set and increasing size — this is the most direct test:
   ping -M do -s 1472 <destination>      # Linux, tests 1500 MTU (1472 + 28 header bytes)
   ping -f -l 1472 <destination>         # Windows

Drop the size (try 1400, 1300, etc.) until it succeeds. If a large ping with DF fails/times out but a smaller one succeeds, and you get no ICMP error back, that's the signature of a black hole. If ICMP worked properly you'd see a "Frag needed" message rather than a silent timeout.

traceroute/tracepath with varying packet sizes — tracepath on Linux auto-discovers PMTU and reports it directly; mtr or traceroute with size options can help isolate which hop is problematic.
Check TCP handshake vs data transfer behavior — if the SYN/SYN-ACK/ACK completes but the connection stalls right after, especially under load or with larger requests, that's a strong PMTUD black hole indicator, since the handshake packets are small.
Packet capture (tcpdump/Wireshark) — capture on both ends. Look for: retransmissions of the same full-size segment, no incoming ICMP Type 3 Code 4 messages, and stalling right as segment sizes increase (often visible right after the initial small handshake/ACK packets).
Check for ICMP filtering directly — verify whether firewalls in the path (yours or upstream) block ICMP Type 3 (Destination Unreachable) messages, since Code 4 (Fragmentation Needed) is a subtype of Type 3. Many firewall configs block ICMP wholesale without carving out this exception.

- What's TCP MSS clamping and where is it usually configured?

TCP MSS Clamping. MSS (Maximum Segment Size) is the largest chunk of TCP payload data a host will accept in a single segment. It's negotiated during the TCP three-way handshake, with each side advertising its own MSS based on its outgoing interface MTU (typically MTU minus 40 bytes for IPv4/TCP headers, so 1460 for a standard 1500-byte MTU).

The problem MSS clamping solves:

Sometimes a link in the path has a smaller MTU than either endpoint's interface — the classic case being PPPoE (used by many DSL/fiber ISPs), which adds 8 bytes of overhead, dropping usable MTU to 1492 instead of 1500. If both endpoints negotiate an MSS based on 1500, their packets will be too large to cross that link.

Normally, Path MTU Discovery (PMTUD) handles this: routers send back an ICMP "Fragmentation Needed" message and the sender reduces its packet size. But PMTUD often breaks in practice because:

Firewalls block ICMP entirely (a common, if misguided, security practice)
The ICMP message gets lost or filtered somewhere in the path
Some middleboxes just drop oversized packets silently

When PMTUD fails, you get the classic symptom: small packets (like pings, DNS lookups, SSH sessions) work fine, but anything requiring full-size packets (loading web pages, large file transfers) hangs or times out. This happens because the TCP handshake itself succeeds (small packets), but data transfer with full-size segments stalls.

What MSS clamping does:

Instead of relying on ICMP, a router or firewall directly rewrites the MSS value inside the TCP SYN and SYN-ACK packets as they pass through, forcing both endpoints to negotiate a smaller MSS that matches the actual constrained link — proactively avoiding the need for fragmentation or PMTUD at all.

Where it's typically configured:

Routers/gateways at the WAN edge — especially on PPPoE links (DSL modems/routers, edge routers terminating PPPoE)
Firewalls — pfSense, OPNsense, iptables/netfilter (--clamp-mss-to-pmtu or explicit --set-mss), Cisco IOS (ip tcp adjust-mss)
VPN gateways — IPsec, WireGuard, OpenVPN concentrators, since tunnel encapsulation overhead (headers, encryption) reduces the effective MTU, and VPN traffic often has "don't fragment" flags set that make PMTUD unreliable
Cloud/virtual network edges — some cloud load balancers or NAT gateways apply it automatically for tunneled or overlay networks (e.g., VXLAN, GRE)

The general rule of thumb: clamp MSS on the device that sits at the boundary between the full-MTU network and the reduced-MTU link, applying it to traffic in both directions so both the client's and server's segments get adjusted.