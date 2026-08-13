# `tcpdump`: packet evidence

`tcpdump` answers “what crossed this capture point?” A useful capture has:

1. the correct namespace;
2. the correct interface;
3. a narrow filter;
4. a stopping condition.

Capturing `any` with no filter produces noise and weakens your reasoning.

## Capture point first

In the routed lab, the same flow can be observed at four useful points:

```text
client c-eth0 -> router r-left -> router r-right -> server s-eth0
```

Capture on both router interfaces when you suspect forwarding or NAT. If a
packet enters `r-left` but does not leave `r-right`, the boundary is the router.
If it leaves but no reply returns, move the observation to the server.

`-i any` is good for discovery but does not preserve a normal Ethernet view on
Linux. Switch to a specific interface once you know the path.

## Option anatomy

```text
tcpdump [capture options] [BPF filter]

-i IFACE   capture interface       -n no name resolution
-e         show L2 header          -v/-vv more decode detail
-c COUNT   stop after count        -w FILE write raw pcap
-r FILE    read pcap               -A print ASCII payload
-X         hex and ASCII           -s 0 capture full packet
```

Default to `-ni`, then add one output or stopping option:

```bash
tcpdump -ni c-eth0 -c 10 icmp
tcpdump -nei r-left -c 10 arp
tcpdump -ni s-eth0 -w /tmp/http.pcap 'tcp port 8080'
tcpdump -nnr /tmp/http.pcap
```

Use `-nn` when service-name resolution would turn port numbers into names.
Never use `-A` on unknown or sensitive production traffic without considering
credentials and personal data.

## BPF filter anatomy

Build filters from primitives:

```text
direction + kind + value + boolean composition
```

Examples:

```bash
host 10.20.2.10
src host 10.20.1.10
dst net 10.20.2.0/24
tcp port 8080
src portrange 1024-65535
icmp
arp
'host 10.20.2.10 and (icmp or tcp port 8080)'
'tcp port 8080 and not host 10.20.1.99'
```

Shell metacharacters make quoting the complete filter a good habit.

TCP flag test:

```bash
'tcp port 8080 and tcp[tcpflags] & (tcp-syn|tcp-fin|tcp-rst) != 0'
```

The three-way handshake should appear as:

```text
client:ephemeral > server:8080 Flags [S]
server:8080 > client:ephemeral Flags [S.]
client:ephemeral > server:8080 Flags [.]
```

Repeated `[S]` with no `[S.]` means the client did not receive a SYN-ACK. It
does not tell you whether the SYN or the return packet was lost; compare
capture points.

## Packet fields to read

For each line, say aloud:

- timestamp;
- source and destination;
- protocol and ports;
- flags;
- sequence/acknowledgment relationship;
- length;
- IP TTL when visible.

With `-e`, also read source/destination MAC and EtherType. Routers rewrite L2
headers on each link while preserving end-to-end IPs unless NAT applies. TTL
decreases at the router.

## Muscle-memory drill

```bash
./scripts/setup-routed.sh
./scripts/setup-services.sh
```

Use two shells. Start each capture first, then generate exactly one request:

1. Capture only ARP on the client link. Flush the client's neighbor cache and
   ping the server. Identify who asks and who answers.
2. Capture exactly four ICMP packets on `r-left`.
3. Capture TCP/8080 handshake/close flags on `r-right`.
4. Write one HTTP flow to `/tmp/http.pcap`, read it numerically, then print its
   ASCII payload.
5. Capture simultaneously on `r-left` and `r-right`. Compare MAC addresses,
   TTL, and IP addresses.

Useful traffic generators:

```bash
ip netns exec clp-client ping -c 2 10.20.2.10
ip netns exec clp-client curl -s http://10.20.2.10:8080/
```

## Diagnose, do not guess

For an attempted TCP connection, complete this matrix in your notes:

```text
point       SYN seen   SYN-ACK seen   inference
client
r-left
r-right
server
```

The first adjacent pair whose observations differ identifies the failing
boundary.

## Retrieval test

Without notes, capture:

- five packets to or from one host, with no name resolution;
- ARP including Ethernet headers;
- only TCP SYN/RST/FIN for port 8080;
- a full pcap and then read it without resolving hostnames or ports;
- traffic for two ports while excluding one host.
