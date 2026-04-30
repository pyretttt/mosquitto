# Curated References

Bookmark this. Each section is ordered roughly from "first read" to "deep dive".

## Primary textbook

- Kurose & Ross, *Computer Networking: A Top-Down Approach*, 8th edition. The book this whole plan is built around.
- Companion site (slides, Wireshark labs, programming assignments, errata): http://gaia.cs.umass.edu/kurose_ross/
- Authors' Wireshark labs (free PDFs + .pcap files): http://gaia.cs.umass.edu/kurose_ross/wireshark.php

## Reference books to dip into

- W. Richard Stevens & Kevin Fall, *TCP/IP Illustrated, Vol. 1: The Protocols*, 2nd ed. - the canonical low-level reference.
- Christian Benvenuti, *Understanding Linux Network Internals*. Old but the kernel concepts haven't changed much.
- Cricket Liu & Paul Albitz, *DNS and BIND* - the DNS reference.
- Iljitsch van Beijnum, *BGP*. Short and focused.
- Lee Calcote & Nic Jackson, *Container Networking* (free O'Reilly short).

## Beginner-friendly explainers

- Beej's Guide to Network Programming: https://beej.us/guide/bgnet/
- Julia Evans zines and posts (highly recommended):
  - "How DNS works" https://wizardzines.com/zines/dns/
  - "tcpdump" https://wizardzines.com/zines/tcpdump/
  - "HTTP: Learn your browser's language" https://wizardzines.com/zines/http/
  - "Networking, ACK!" https://wizardzines.com/zines/networking/
- "An Illustrated Guide to OAuth and OpenID Connect" - useful tangent.
- "High Performance Browser Networking" by Ilya Grigorik (free online): https://hpbn.co/

## RFCs you should at least skim

The book references most of these. Read the introduction and the section that maps to whatever you're confused about.

- RFC 791 - IP (1981)
- RFC 792 - ICMP (1981)
- RFC 768 - UDP (1980)
- RFC 793 / RFC 9293 - TCP (original / consolidated 2022)
- RFC 1034 / 1035 - DNS concepts and specification
- RFC 2131 - DHCP
- RFC 4271 - BGP-4
- RFC 5246 - TLS 1.2 (skim, then jump to 8446)
- RFC 8446 - TLS 1.3
- RFC 9000 - QUIC
- RFC 9110 - HTTP semantics
- RFC 9112 - HTTP/1.1
- RFC 9113 - HTTP/2
- RFC 9114 - HTTP/3
- RFC 4443 - ICMPv6
- RFC 8200 - IPv6
- RFC 7348 - VXLAN
- RFC 4301 - IPsec architecture

## Linux network stack

- LWN.net networking articles (search for "networking" tag): https://lwn.net/Kernel/Index/#Networking
- "Linux Networking Stack" (kernel docs): https://docs.kernel.org/networking/
- Brendan Gregg, "Linux Network Performance Tools": https://www.brendangregg.com/linuxperf.html
- "Monitoring and Tuning the Linux Networking Stack" by Joe Damato:
  - Receiving: https://blog.packagecloud.io/monitoring-tuning-linux-networking-stack-receiving-data/
  - Sending: https://blog.packagecloud.io/monitoring-tuning-linux-networking-stack-sending-data/
- iproute2 cookbook: https://baturin.org/docs/iproute2/

## TCP / congestion control / QUIC

- "TCP/IP Illustrated, Vol 1" - chapters 13-19.
- "BBR: Congestion-Based Congestion Control" (Cardwell et al., ACM Queue 2016): https://queue.acm.org/detail.cfm?id=3022184
- Cloudflare blog tag "TCP": https://blog.cloudflare.com/tag/tcp/
- Cloudflare on QUIC and HTTP/3: https://blog.cloudflare.com/tag/quic/
- "Making the QUIC handshake faster" (Cloudflare): https://blog.cloudflare.com/optimizing-tls-over-tcp-to-reduce-latency/

## DNS

- Julia Evans "Implement DNS in a weekend": https://implement-dns.wizardzines.com/
- Cloudflare 1.1.1.1 docs and blog posts on DoH/DoT.
- DNSViz for visualizing real-world DNS chains: https://dnsviz.net

## TLS / crypto

- "The Illustrated TLS 1.3 Connection": https://tls13.xargs.org/
- "The Illustrated TLS 1.2 Connection": https://tls12.xargs.org/
- BetterTLS test suite: https://bettertls.com/
- Mozilla SSL Configuration Generator: https://ssl-config.mozilla.org/
- testssl.sh: https://testssl.sh/

## Firewalls / netfilter

- nftables wiki: https://wiki.nftables.org/wiki-nftables/index.php/Main_Page
- "A Deep Dive into Iptables and Netfilter Architecture" (DigitalOcean): https://www.digitalocean.com/community/tutorials/a-deep-dive-into-iptables-and-netfilter-architecture
- conntrack-tools: https://conntrack-tools.netfilter.org/

## Kubernetes networking

- Official docs: https://kubernetes.io/docs/concepts/services-networking/
- Cilium docs (CNI + eBPF): https://docs.cilium.io
- "A visual guide to Kubernetes networking fundamentals" (Amim Knabben, LWN): https://lwn.net/Articles/831087/
- Learnk8s networking series: https://learnk8s.io/kubernetes-network-packets
- "Life of a Packet" series by Iman Tumorang and others on Medium.
- "Cilium - eBPF-based Networking, Observability, and Security" (KubeCon talks): YouTube search.

## AWS networking

- VPC user guide: https://docs.aws.amazon.com/vpc/latest/userguide/
- "Networking Fundamentals" whitepaper: https://docs.aws.amazon.com/whitepapers/latest/aws-vpc-connectivity-options/welcome.html
- re:Invent NET400 "Advanced VPC Design" (search YouTube for the latest year)
- "A Day in the Life of a Billion Packets" (re:Invent classic).
- VPC Reachability Analyzer docs: https://docs.aws.amazon.com/vpc/latest/reachability/
- Workshop: "Networking Workshop for Builders" https://catalog.workshops.aws/networking/

## Tooling

- `tcpdump` man page + the cheat sheet at https://danielmiessler.com/study/tcpdump/
- Wireshark User Guide: https://www.wireshark.org/docs/wsug_html_chunked/
- `ss` man page + `man tc` (note: `tc` is poorly documented; LARTC HOWTO is still the best: https://lartc.org/howto/)
- mtr: https://www.bitwizard.nl/mtr/
- iperf3: https://iperf.fr/
- netshoot toolbox image: https://github.com/nicolaka/netshoot

## Talks worth your time

- "It's Always DNS" - any year of any conference.
- "BPF: A New Type of Software" - Brendan Gregg.
- "Linux Networking Demystified" - LISA / SREcon talks.
- re:Invent NET-tagged sessions, latest year.

## Russian-language supplements (optional)

- Олиферы, "Компьютерные сети. Принципы, технологии, протоколы" - more depth on physical layer if you want it.
- Cisco / Linux Academy translations on Stepik.
- Habr "Сетевые технологии" hub for war stories.

If you find a great resource not listed here, add it. This file is meant to grow.
