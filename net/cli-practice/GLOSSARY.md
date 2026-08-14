# Glossary

Short definitions for terms this track actually uses. Read a definition, then
go to the module that teaches it. Do not try to memorize this page first.

## How to use this page

If a module says "see the glossary," read **one** entry and return. If an
entry is still opaque, you are probably reading a later module too early.

## Addresses and layers

- **Layer / L2 / L3 / L4** — A job in the stack, not a physical shelf. L2
  delivers a frame to the next machine on the same cable (MAC). L3 delivers a
  packet toward an IP destination, possibly across routers. L4 delivers bytes
  to a process (port). Taught in [foundations](00-foundations.md).
- **Frame** — L2 container: Ethernet header (MACs, type) plus payload.
- **Packet** — L3 container: IP header (IPs, TTL, protocol) plus payload.
  People also say "packet" loosely for the whole thing; this track says
  **frame** when MACs matter and **packet** when IPs matter.
- **Segment** — L4 container for TCP (ports, flags, sequence numbers).
- **MAC address** — 48-bit L2 identity of one interface, written
  `aa:bb:cc:dd:ee:ff`. Meaningful only on the local cable.
- **IP address** — L3 identity. This track uses IPv4, four decimal numbers
  `0–255`, for example `10.20.1.10`.
- **Port** — 16-bit L4 number that selects a process on a host. `8080` is
  the lab HTTP service; `5201` is iperf3.
- **CIDR / prefix** — `10.20.1.0/24` means "addresses whose first 24 bits
  match." `/24` is a 256-address neighborhood; `/32` is one host; `/0` is
  the default route (everything).
- **Subnet** — The set of addresses that share a prefix and can usually
  reach each other without a router.
- **Host vs network vs broadcast** — In `10.20.1.0/24`, `.0` names the
  subnet, `.255` is the broadcast address, `.1`–`.254` can be hosts. The
  lab uses `.1` for router interfaces and `.10` for endpoints.
- **Default gateway** — The IP a host sends to when the destination is off
  its subnet. Client gateway: `10.20.1.1`. Server gateway: `10.20.2.1`.
- **TTL** — Hop budget in the IP header. Each router subtracts 1. At 0 the
  packet is dropped and an ICMP error may be sent. `mtr` uses this on
  purpose.

## Forwarding and neighbors

- **Interface / link / device** — A NIC, or a virtual stand-in (`c-eth0`,
  `r-left`). `ip link` shows it; `ip addr` shows addresses on it.
- **On-link** — A destination the kernel believes is on the same cable as
  an interface, so it will ARP for it instead of sending to a gateway.
- **Route** — A mapping: destination prefix → outgoing interface and,
  usually, next-hop IP. Taught in [`ip`](02-ip.md).
- **Longest prefix match** — When several routes match, the most specific
  prefix wins (`/32` beats `/24` beats `/0`).
- **Next hop** — The neighbor that should receive the frame for this packet.
- **Neighbor / ARP** — Mapping from on-link IP to MAC. ARP asks "who has
  this IP?" on Ethernet. NDP is the IPv6 equivalent; this track does not
  use it.
- **Forwarding** — A router receiving a packet not addressed to itself, then
  sending it out another interface. Requires `net.ipv4.ip_forward=1`.
- **Asymmetric routing** — Request and reply take different paths. Easy to
  misread if you capture only one direction.

## Transport and sockets

- **Socket** — Kernel object identified, for TCP, by a **four-tuple**:
  local IP, local port, remote IP, remote port. Taught in [`ss`](03-ss.md).
- **Listener** — A socket in `LISTEN`, waiting for new connections on a
  local address and port.
- **Ephemeral port** — The short-lived client port picked from a high range.
- **TCP handshake** — `SYN`, `SYN-ACK`, `ACK`. Connection exists only after
  all three. See [foundations](00-foundations.md) and
  [`tcpdump`](04-tcpdump.md).
- **TCP states** — `LISTEN`, `SYN-SENT`, `SYN-RECV`, `ESTABLISHED`,
  `TIME-WAIT`, and others. `ss` prints a short form (`ESTAB`, `TIME-WAIT`).
- **UDP** — Datagram transport with no handshake. The lab uses it mainly in
  iperf3 tests.
- **ICMP** — Control messages, including ping (echo request/reply) and
  "time exceeded" used by traceroute/`mtr`. Ping success does not imply
  TCP success.
- **MSS** — Largest TCP payload the sender will emit. Related to MTU.
- **MTU** — Largest L3 payload an interface will send. Ethernet default is
  1500 bytes.
- **RTT** — Round-trip time.
- **BDP** — Bandwidth-delay product: how much data must be in flight to fill
  the path. Taught in [`mtr` / `iperf3`](05-mtr-iperf3.md).
- **cwnd** — Congestion window: the sender's current cap on unacked data.

## Namespaces and the lab

- **Network namespace** — A Linux isolation boundary for interfaces,
  addresses, routes, and firewall rules. `clp-client`, `clp-router`, and
  `clp-server` are namespaces, not Docker containers.
- **veth pair** — Two virtual interfaces joined like a cable: a frame out
  one end appears in the other.
- **`ip netns exec NAME COMMAND`** — Run `COMMAND` as if you were logged
  into namespace `NAME`. This is how you "sit on" the client, router, or
  server.
- **Observation point** — The namespace **and** interface where you look.
  The same flow looks different on `r-left` and `r-right`.

## Netfilter, firewalls, and NAT

- **Netfilter** — Kernel packet hooks. `iptables` and `nft` are two
  user-space languages for the same machinery.
- **Hook** — A moment in the packet path: prerouting, input, forward,
  output, postrouting.
- **Chain** — An ordered list of rules attached to a hook (or jumped to).
- **Policy** — What happens if no rule matches (`ACCEPT` or `DROP`).
- **Stateful firewall** — Allows return traffic because the kernel
  remembers the flow. Depends on **conntrack**.
- **Stateless firewall** — Each packet is judged alone.
- **SNAT** — Rewrite source address (and maybe port) as the packet leaves.
  Classic "private host uses the router's public IP."
- **DNAT** — Rewrite destination as the packet arrives. Classic "publish
  an internal port on the router's address."
- **MASQUERADE** — SNAT that takes the current outgoing interface address.
- **Conntrack** — The kernel table of flows. NAT and stateful rules both
  read it. Taught in [conntrack](08-conntrack-nat.md).
- **NEW / ESTABLISHED / RELATED / INVALID** — Conntrack classes for a
  packet, not the same thing as TCP's `ESTABLISHED` state.

## Performance

- **Latency** — How long one packet takes. `mtr` is the first tool.
- **Throughput / capacity** — How many bits per second a path can carry
  under a given test. `iperf3` is the first tool.
- **Loss** — Packets that never arrive. Intermediate `mtr` loss is often a
  lie; destination loss is the one that counts.
- **Jitter** — Variation in delay, reported by UDP iperf3.
- **qdisc** — Queuing discipline on an interface, including lab impairments
  (`netem`). Egress-only: it shapes packets **leaving** that interface.

## Possible gap

This glossary does not define BGP, OSPF, VXLAN, TLS, DNS records, or
Kubernetes objects. Those are real topics. They are not on the critical path
for finishing this lab. If a production incident involves them, you still
start with interfaces, routes, sockets, captures, and policy — the same
chain this track drills.
