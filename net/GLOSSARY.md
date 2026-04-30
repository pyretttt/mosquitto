# Glossary

Quick lookup for terms used throughout the curriculum. When you forget what something means, grep here first. Items are intentionally short - go to the relevant week's `notes.md` or the book for depth.

## Layers and addressing

- **L1 / Physical layer** - Bits on a wire. We mostly skim this.
- **L2 / Link layer** - Frames between directly connected nodes. MAC addresses, Ethernet, ARP, switches, VLANs.
- **L3 / Network layer** - Packets across networks. IP addresses, routing, ICMP, NAT.
- **L4 / Transport layer** - End-to-end conversations. UDP (best effort), TCP (reliable), QUIC (TCP+TLS reimagined on UDP).
- **L7 / Application layer** - HTTP, DNS, SMTP, gRPC, etc. The book lumps L5-L7 here.
- **MAC address** - 48-bit unique-per-NIC L2 address (e.g. `aa:bb:cc:dd:ee:ff`).
- **IP address** - 32-bit IPv4 or 128-bit IPv6 L3 address.
- **CIDR** - `10.0.0.0/24` notation. `/24` = 24 high bits are the network, 8 low bits are the host (256 addresses).
- **Subnet** - A contiguous range of IPs that share a CIDR prefix.
- **Default gateway** - The IP a host sends a packet to when the destination is outside its directly-connected subnet.

## Routing and switching

- **Switch** - L2 device. Forwards frames based on MAC table. Builds the table by snooping source MACs.
- **Router** - L3 device. Forwards packets based on destination IP using a routing table (longest prefix match).
- **Routing table** - Mapping from destination prefix to (next-hop, outgoing interface).
- **Longest prefix match** - When multiple routes match, the one with the most specific prefix wins.
- **AS (Autonomous System)** - A network under a single administrative authority. Identified by an ASN.
- **IGP (Interior Gateway Protocol)** - Routes inside one AS. OSPF, IS-IS, RIP.
- **EGP (Exterior Gateway Protocol)** - Routes between ASes. BGP-4 is the only one in production.
- **Link-state vs distance-vector** - LS (OSPF) floods topology, each router computes Dijkstra. DV (RIP) gossips distances, vulnerable to count-to-infinity.
- **BGP attributes** - LOCAL_PREF, AS_PATH, MED, ORIGIN, NEXT_HOP. Selection order matters.
- **eBGP / iBGP** - BGP between different ASes / inside the same AS.

## Transport

- **MTU** - Maximum Transmission Unit. Largest L2 frame payload. Ethernet default 1500.
- **MSS** - Maximum Segment Size. Largest TCP payload. Usually MTU - 40 (IPv4+TCP headers) = 1460.
- **PMTUD** - Path MTU Discovery. Sender sets DF bit; routers return ICMP "Frag Needed". Black-holes if ICMP is dropped.
- **MSS clamping** - Router rewrites the MSS option in the SYN to avoid PMTUD black holes (`-j TCPMSS --clamp-mss-to-pmtu`).
- **RTT** - Round-trip time.
- **BDP** - Bandwidth-Delay Product. RTT * bandwidth = ideal in-flight bytes. Window must be >= BDP for full throughput.
- **cwnd / rwnd** - Congestion window (sender's self-imposed cap) / receive window (advertised by peer).
- **SACK** - Selective ACK. Lets the receiver tell the sender exactly which segments arrived.
- **Slow start / congestion avoidance** - cwnd doubles per RTT, then increases linearly above ssthresh.
- **Fast retransmit / fast recovery** - 3 dup ACKs trigger immediate retransmit without waiting for RTO.
- **Reno / CUBIC / BBR** - Congestion control algorithms. CUBIC is default on Linux; BBR models bandwidth and RTT.
- **QUIC** - L4-ish protocol on UDP. Combines TCP+TLS+multiplexing. Faster handshake, 0-RTT, no head-of-line blocking across streams.

## NAT and address translation

- **SNAT** - Source NAT. Rewrites source IP/port (e.g. masquerade behind a public IP).
- **DNAT** - Destination NAT. Rewrites destination (e.g. port forwarding, k8s ClusterIP).
- **MASQUERADE** - SNAT but the source IP is picked from the outgoing interface dynamically.
- **Conntrack** - Kernel connection tracking. Required for stateful NAT and stateful firewalling.
- **Hairpin NAT** - Two clients behind the same NAT trying to reach each other via the NAT's public IP. Often broken without explicit support.

## Application layer

- **HTTP/1.1** - Text protocol, persistent connections, head-of-line blocking per connection.
- **HTTP/2** - Binary, multiplexed streams over one TCP connection, HPACK header compression.
- **HTTP/3** - HTTP over QUIC. No HoL blocking across streams (QUIC handles it).
- **TLS** - Transport Layer Security. 1.2 = 2-RTT handshake; 1.3 = 1-RTT (or 0-RTT with PSK).
- **SNI** - Server Name Indication. TLS extension that lets a server pick a cert based on hostname; sent in cleartext (until ECH).
- **DNS resolver vs authoritative** - Resolver asks; authoritative answers for a zone it owns.
- **DNS record types** - A, AAAA, CNAME, MX, TXT, NS, SOA, SRV, PTR.
- **CoreDNS** - The DNS server inside Kubernetes.

## Security

- **SG (Security Group)** - Stateful AWS firewall attached to ENIs. Implicit deny; rules are allow-only.
- **NACL** - Stateless AWS firewall attached to subnets. Explicit allow + explicit deny, evaluated in order.
- **Stateful firewall** - Tracks connections; allows return traffic automatically. nftables `ct state established,related accept`.
- **Stateless firewall** - Each packet evaluated independently. Faster but rules are more verbose.
- **IPsec** - L3 tunneling (ESP) or authentication (AH). Used for site-to-site VPNs.
- **WireGuard** - Modern, simpler L3 VPN. Public-key based.

## Kubernetes networking

- **Pod IP** - Each pod gets a routable IP (within the cluster). No NAT pod-to-pod.
- **Service / ClusterIP** - Virtual IP that load-balances to a set of pod IPs. Implemented by kube-proxy (iptables/IPVS) or by CNI (eBPF in Cilium).
- **EndpointSlice** - The list of pod IPs backing a Service.
- **kube-proxy** - Watches Services and writes iptables/IPVS rules on each node.
- **CNI (Container Network Interface)** - Plugin contract for assigning pod IPs and connecting them. Cilium, Calico, Flannel.
- **NetworkPolicy** - Pod-level firewall rules. Enforced by the CNI.
- **Ingress / Gateway API** - L7 entry point. Usually nginx, Envoy, or a managed LB.
- **CoreDNS** - In-cluster DNS server. Resolves `service.namespace.svc.cluster.local`.
- **DaemonSet** - Pod scheduled on every node. CNI agents are usually DaemonSets.

## AWS networking

- **VPC** - Your private network in AWS. CIDR-bound, region-scoped.
- **Subnet** - A slice of the VPC's CIDR pinned to one AZ.
- **IGW (Internet Gateway)** - Lets a public subnet talk to the internet.
- **NAT GW** - Lets a private subnet talk to the internet (egress only).
- **Route Table** - Per-subnet (or main) table mapping destination prefix to a target (IGW, NAT GW, ENI, peering, TGW).
- **ENI (Elastic Network Interface)** - A NIC attached to instances/pods/services.
- **ALB / NLB** - L7 / L4 managed load balancers.
- **TGW (Transit Gateway)** - Hub-and-spoke connectivity between VPCs and on-prem.
- **VPC Endpoint** - Reach AWS services (S3, ECR, STS, etc.) without going through the internet/NAT GW. Gateway endpoints (S3, DynamoDB) are free; Interface endpoints cost per-AZ-per-hour.
- **Reachability Analyzer** - Symbolic checker that explains "can A reach B" given your VPC config.
- **Flow Logs** - Per-ENI 5-tuple log with action ACCEPT/REJECT.

## Tools you should know cold

- `ip` - Modern replacement for `ifconfig`, `route`, `arp`. See `cheatsheets/iproute2.md`.
- `ss` - Modern replacement for `netstat`. `ss -tnlp`, `ss -ti`.
- `tcpdump` - Capture packets. See `cheatsheets/tcpdump-wireshark.md`.
- `dig` - DNS lookups. See `cheatsheets/dns-tools.md`.
- `mtr` - traceroute + ping continuous.
- `iperf3` - Throughput testing.
- `nft` - nftables. See `cheatsheets/netfilter.md`.
- `conntrack` - Inspect conntrack table.
- `openssl s_client` - Probe TLS endpoints.
- `kubectl` + `cilium` - Kubernetes networking.
