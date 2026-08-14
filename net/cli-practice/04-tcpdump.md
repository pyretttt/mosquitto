# `tcpdump`: packet evidence

`tcpdump` answers "what crossed this capture point?" Configuration and
sockets tell you what the kernel **intends**. A capture tells you what
**happened on the cable**. Those disagree more often than beginners expect.

## Where you are

You can read addresses, routes, and listeners. This module is how you prove
a packet existed at a place in the path, and how you read the handshake you
met in foundations.

## What you need, and what you do not

Need: encapsulation and ARP from [foundations](00-foundations.md); the idea
of an observation point from the
[debugging method](01-debugging-method.md); [`ip`](02-ip.md) and
[`ss`](03-ss.md).

Do not need: Wireshark, deep IP header bitfields, or writing BPF from
scratch. You need a small set of predicates and the discipline to capture
narrowly.

## Capture point first

A useful capture has:

1. the correct namespace;
2. the correct interface;
3. a narrow filter;
4. a stopping condition.

Capturing `any` with no filter produces noise and weakens your reasoning.

In the routed lab, the same flow can be observed at four useful points:

```text
client c-eth0 → router r-left → router r-right → server s-eth0
```

Capture on both router interfaces when you suspect forwarding or NAT. If a
packet enters `r-left` but does not leave `r-right`, the boundary is the
router. If it leaves but no reply returns, move the observation to the
server.

`-i any` is good for discovery but does not preserve a normal Ethernet view
on Linux. Switch to a specific interface once you know the path.

Run tcpdump **in the namespace that owns the interface**:

```bash
ip netns exec clp-router tcpdump -ni r-left -c 10 icmp
```

## What you are looking at

Each printed line is a decoded packet. For IP traffic, read in this order:

1. timestamp;
2. source and destination (IP, plus ports for TCP/UDP);
3. protocol and flags;
4. length;
5. TTL when visible (`-v`).

With `-e`, also read source/destination MAC and EtherType. Routers rewrite
L2 headers on each link while preserving end-to-end IPs unless NAT
applies. TTL decreases at the router. If IPs change between `r-left` and
`r-right`, you are looking at NAT, not ordinary forwarding.

The three-way handshake should appear as:

```text
client:ephemeral > server:8080 Flags [S]
server:8080 > client:ephemeral Flags [S.]
client:ephemeral > server:8080 Flags [.]
```

`[S]` is SYN. `[S.]` is SYN plus ACK. `[.]` is ACK without SYN/FIN/RST.
Repeated `[S]` with no `[S.]` means the client did not receive a SYN-ACK.
It does not tell you whether the SYN or the return packet was lost;
compare capture points.

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
Never use `-A` on unknown or sensitive production traffic without
considering credentials and personal data. In this lab the HTTP payload is
a boring directory listing; `-A` is safe and instructive.

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

A filter is a predicate on **this packet at this capture point**. After
NAT, `host 10.30.1.10` may match on the inside and fail on the outside.
That disagreement is the evidence, not a broken filter.

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
boundary. Fill it with captures, not with imagination.

## Common misconceptions

- **"I captured `any`, so I cannot have missed it."** You can still be in
  the wrong namespace. You can still filter the peer address after NAT.
  You can still start tcpdump **after** the SYN already happened.
- **"No packets means drop."** Empty capture plus failed `route get`
  means the host never sent. Empty capture plus a wrong filter means you
  hid the packets from yourself.
- **"ARP is noise."** ARP is how the next hop is found. Filter it on
  purpose when neighbors are the question.

### Possible gap: offloads, checksums, and truncated packets

NICs (and some virtual NICs) compute checksums in hardware. tcpdump may
print "bad checksum" on packets that the stack will accept. Do not chase
that on this lab. `-s 0` captures full packets; the default snaplen can
truncate payloads. For handshake flags, default snaplen is enough.

## Muscle-memory drill

```bash
./scripts/setup-routed.sh
./scripts/setup-services.sh
```

Use two shells. Start each capture first, then generate exactly one
request:

1. Capture only ARP on the client link. Flush the client's neighbor cache
   and ping the server. Identify who asks and who answers.
2. Capture exactly four ICMP packets on `r-left`.
3. Capture TCP/8080 handshake/close flags on `r-right`.
4. Write one HTTP flow to `/tmp/http.pcap`, read it numerically, then print
   its ASCII payload.
5. Capture simultaneously on `r-left` and `r-right`. Compare MAC addresses,
   TTL, and IP addresses.

Useful traffic generators:

```bash
ip netns exec clp-client ping -c 2 10.20.2.10
ip netns exec clp-client curl -s http://10.20.2.10:8080/
```

Neighbor cache flush, when you need a fresh ARP:

```bash
ip -n clp-client neigh flush all
```

## Retrieval test

Without notes, capture:

- five packets to or from one host, with no name resolution;
- ARP including Ethernet headers;
- only TCP SYN/RST/FIN for port 8080;
- a full pcap and then read it without resolving hostnames or ports;
- traffic for two ports while excluding one host.

Next: [`mtr` and `iperf3`](05-mtr-iperf3.md).
