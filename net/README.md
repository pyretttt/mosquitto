# net - DevOps Networking Learning Project

A 16-week, ~5-7 hrs/week curriculum that re-reads Kurose & Ross "Computer Networking: A Top-Down Approach" (8th ed.) and pairs each chapter with hands-on labs in a hybrid environment (Linux network namespaces + Docker + kind/k3d + AWS Free Tier).

## Goals

By the end you will be able to:

1. Explain L2-L7 protocols and trade-offs without notes.
2. Bring up a multi-node lab and reproduce real production failure modes (PMTUD black-hole, NAT hairpin, BGP path selection, conntrack overflow, TLS handshake debug, etc.).
3. Debug "the network" with `tcpdump`, `ss`, `ip`, `dig`, `mtr`, `nft`, `conntrack`, `iperf3`, `openssl s_client` from muscle memory.
4. Reason about Kubernetes pod-to-pod, pod-to-service, and ingress paths end-to-end.
5. Design and deploy a hardened multi-AZ AWS VPC with EKS, ALB, R53, and VPC endpoints, then trace a request from your laptop to a database row.

## How to use this repo

1. Read [LEARNING_PLAN.md](LEARNING_PLAN.md) for the full schedule.
2. Each week has a folder `weekNN-topic/` with its own `README.md`. Open the current week's README, follow the reading list, run the lab, do the exercises, write notes in `notes.md`.
3. `cheatsheets/` are reference cards. Skim them once at the start, return to them when you forget a flag.
4. `lab/` holds reusable artifacts (netns scripts, docker-compose stacks, kind config, terraform). Each week's lab section tells you which to use.
5. `capstone/` is the W16 deliverable.

## Quick start

```bash
cd net

cat lab/netns/SETUP.md

bash lab/netns/two-hosts.sh
sudo ip netns exec h1 ping -c 3 10.0.0.2
bash lab/netns/teardown.sh
```

## Conventions

- Every command is shown with the directory you should run it in.
- pcap files captured during labs go in `weekNN-*/captures/` (gitignored).
- Free-form notes in `weekNN-*/notes.md` (gitignored by default; remove from `.gitignore` if you want to commit them).
- AWS artifacts cost money. The capstone has a tear-down script. Do `terraform destroy` before stopping for the day.

## Status tracker

Tick a box when you finish a week. Use commits as a journal.

- [ ] W1  Ch1 part 1 + lab bootstrap
- [ ] W2  Ch1 part 2 + Wireshark intro
- [ ] W3  Ch2 HTTP/Web
- [ ] W4  Ch2 DNS + sockets
- [ ] W5  Ch3 UDP + RDT
- [ ] W6  Ch3 TCP + congestion + QUIC
- [ ] W7  Ch4 IP forwarding + fragmentation
- [ ] W8  Ch4 NAT + IPv6 + SDN data plane
- [ ] W9  Ch5 OSPF + intra-AS routing
- [ ] W10 Ch5 BGP + ICMP + SDN control plane
- [ ] W11 Ch6 link layer + VLANs (Ch7 skim)
- [ ] W12 Ch8 crypto + TLS
- [ ] W13 Ch8 IPsec + firewalls + netfilter deep dive
- [ ] W14 Kubernetes networking
- [ ] W15 AWS VPC networking
- [ ] W16 Capstone: EKS in hardened VPC
