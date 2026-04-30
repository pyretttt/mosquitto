# 16-Week Learning Plan

Pace: ~5-7 hrs/week. Two weeks per book chapter for Ch1-Ch5; one week each for Ch6/Ch7-skim/Ch8 split; then four weeks for Kubernetes, AWS, and the capstone. Physical layer is intentionally skimmed - L2-L7 plus cloud is where you spend cycles.

For the philosophy and project layout, see [README.md](README.md). For curated links, see [REFERENCES.md](REFERENCES.md). For terms, see [GLOSSARY.md](GLOSSARY.md).

## Schedule at a glance

| Week | Book chapter           | Topic                                     | Folder |
|------|------------------------|-------------------------------------------|--------|
| 1    | 1.1-1.3                | What the Internet is + lab bootstrap      | [week01-introduction-bootstrap](week01-introduction-bootstrap/) |
| 2    | 1.4-1.7                | Delay, loss, throughput + Wireshark intro | [week02-delay-throughput-wireshark](week02-delay-throughput-wireshark/) |
| 3    | 2.1-2.3                | HTTP and the Web                          | [week03-http-web](week03-http-web/) |
| 4    | 2.4-2.7                | DNS + sockets + P2P                       | [week04-dns-sockets](week04-dns-sockets/) |
| 5    | 3.1-3.4                | UDP + reliable transfer principles        | [week05-udp-rdt](week05-udp-rdt/) |
| 6    | 3.5-3.8                | TCP + congestion control + QUIC           | [week06-tcp-congestion-quic](week06-tcp-congestion-quic/) |
| 7    | 4.1-4.3                | Forwarding, IP, MTU, fragmentation        | [week07-ip-forwarding-fragmentation](week07-ip-forwarding-fragmentation/) |
| 8    | 4.4-4.5                | NAT, IPv6, SDN data plane                 | [week08-nat-ipv6-sdn](week08-nat-ipv6-sdn/) |
| 9    | 5.1-5.3                | Intra-AS routing (OSPF), LS vs DV         | [week09-ospf-intra-as](week09-ospf-intra-as/) |
| 10   | 5.4-5.7                | BGP + ICMP + SDN control plane            | [week10-bgp-icmp-sdn](week10-bgp-icmp-sdn/) |
| 11   | 6 + 7-skim             | Link layer, ARP, switches, VLANs          | [week11-link-layer-vlans](week11-link-layer-vlans/) |
| 12   | 8.1-8.5                | Crypto basics + TLS                       | [week12-crypto-tls](week12-crypto-tls/) |
| 13   | 8.6-8.9                | IPsec, firewalls, IDS, netfilter          | [week13-firewall-netfilter](week13-firewall-netfilter/) |
| 14   | (docs)                 | Kubernetes networking                     | [week14-k8s-networking](week14-k8s-networking/) |
| 15   | (docs)                 | AWS VPC + edge                            | [week15-aws-vpc](week15-aws-vpc/) |
| 16   | (capstone)             | EKS in hardened VPC + packet-flow doc     | [week16-capstone-eks](week16-capstone-eks/) |

## Per-week routine

Each week:

1. Read the assigned sections (~30-50 pages) before touching any commands.
2. Bring up the lab and reproduce the chapter's main ideas.
3. Do the exercises in `weekNN/exercises/`.
4. Write your own summary in `weekNN/notes.md` (it's gitignored by default).
5. Answer the self-check questions in your head, in writing, or to a rubber duck.
6. Commit your work with a descriptive message: `wNN: <what you did>`.

If you skip a step, you skip the learning. The book is good but passive; the labs are where production-grade intuition is built.

## What "done" looks like for each week

- You can answer every self-check question without scrolling back.
- Your `notes.md` would help your past-self the next time the topic shows up.
- You can reproduce one specific failure mode the chapter introduced (PMTUD black hole, count-to-infinity, NAT hairpin, conntrack table overflow, TLS expired cert, etc.).

## Capstone

Week 16 produces a single document, `capstone/PACKET_FLOW.md`, that traces a request from your laptop to a row in RDS through every hop, header rewrite, NAT, conntrack entry, SG/NACL evaluation, route-table lookup, and kube-proxy/Cilium decision. If you can write that document without notes, you have the networking foundation a senior DevOps engineer needs.
