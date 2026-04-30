# Week 10 - Ch5 part 2: BGP + ICMP + SDN control plane

BGP is the protocol that holds the Internet together (and famously breaks it). ICMP is the Swiss Army knife for diagnostics. SDN control plane is the model on top of which managed networking (k8s, AWS) is built.

## Reading

- Kurose & Ross 8e, 5.4-5.7 (~40 pages):
  - 5.4 Routing among ISPs: BGP
  - 5.5 The SDN control plane
  - 5.6 ICMP
  - 5.7 Network management, SNMP, NETCONF
- RFC 4271 (BGP-4) - read sections 1-5.
- RFC 7911 (ADD-PATH for BGP) - skim.
- RFC 792 (ICMPv4)
- (optional) "BGP from Scratch" series on Cumulus Networks blog.
- (optional) "Why the BGP outage of <pick a year> happened" - any postmortem (Facebook 2021, Cloudflare 2020, AWS, etc.)

## Learning goals

- Bring up an eBGP session between two ASes in a netns lab and watch a prefix propagate.
- Explain the BGP best-path selection order: weight (Cisco) -> LOCAL_PREF -> AS_PATH length -> origin -> MED -> ...
- Read a real BGP table snippet and explain why a particular path was chosen.
- Know the common ICMP types/codes (Echo, Echo Reply, Destination Unreachable codes 0/1/3/4, Time Exceeded, Redirect).
- Sketch how an SDN controller (OpenFlow / Cilium operator / AWS control plane) differs from a distributed control plane (OSPF, BGP).

## Lab

**Goal:** two ASes peering with eBGP, advertise/withdraw a prefix.

```
   AS65001                AS65002
   r1 (10.10.0.1) <-eBGP-> r2 (10.20.0.1)
       loopback             loopback
       192.0.2.0/24         198.51.100.0/24
```

Use FRR's `bgpd`. Two netns, peering on a /30 link.

```bash
sudo ip netns exec r1 vtysh
# router bgp 65001
# neighbor 10.30.0.2 remote-as 65002
# network 192.0.2.0/24

sudo ip netns exec r2 vtysh
# router bgp 65002
# neighbor 10.30.0.1 remote-as 65001
# network 198.51.100.0/24
```

Capture the BGP session establishment and updates:

```bash
sudo ip netns exec r1 tcpdump -ni any -e tcp port 179 -w /tmp/bgp.pcap
```

Watch updates:

```bash
sudo ip netns exec r1 vtysh -c 'show ip bgp summary'
sudo ip netns exec r1 vtysh -c 'show ip bgp'
```

Withdraw a prefix and watch:

```bash
sudo ip netns exec r2 vtysh
# router bgp 65002
# no network 198.51.100.0/24
```

## Exercises

1. **Conceptual.** In `notes.md`:
   - Walk through the BGP best-path selection algorithm from top to bottom.
   - What's the difference between iBGP and eBGP? Why does iBGP need a full mesh (or route reflectors)?
   - Pick a real BGP postmortem (Facebook Oct 2021 is the canonical one), read the writeup, summarize what BGP did wrong and what the fix was.
2. **Practical.** Run the 2-AS lab. Withdraw / re-advertise a prefix while pinging from r1's loopback to r2's. Note the convergence time.
3. **Practical - ICMP forensics.** Use `traceroute` and `traceroute -I` (UDP probe vs ICMP Echo). Capture both. In Wireshark, identify how each hop is discovered. Trigger Destination Unreachable / Time Exceeded deliberately and see them.
4. **Stretch - real BGP.** Open a free RIPE RIS BGP looking glass (https://stat.ripe.net/), look up your favorite ISP's prefix, see what AS-paths exist. Comment.

## Self-check

- Without notes: list the 5 BGP attributes you'd check first when debugging a path-selection question.
- What's TCP/179, and why does BGP use TCP and not UDP?
- What's an SDN "northbound API" vs "southbound API"?
- What does ICMP type 3 code 4 mean? When does it get sent?

## Useful one-liners

```bash
vtysh -c 'show bgp ipv4 unicast summary'
vtysh -c 'show bgp ipv4 unicast 192.0.2.0/24'
vtysh -c 'show bgp ipv4 unicast neighbors 10.30.0.2 advertised-routes'
vtysh -c 'show bgp ipv4 unicast neighbors 10.30.0.2 received-routes'

mtr -bw 1.1.1.1
traceroute -I 1.1.1.1
traceroute -T -p 443 1.1.1.1
```
