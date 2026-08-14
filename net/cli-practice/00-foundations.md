# Foundations

This module is the theory the rest of the track stands on. You can read it
before entering the lab. After the first half, bring the topology up and look
at the real objects with the commands at the end.

## Where you are

You can start a shell and run a command. You have not been asked to know
what a subnet, a MAC address, or a TCP handshake is. By the end of this
page you should be able to walk one ping from `clp-client` to `clp-server`
in words, then confirm the picture with `ip`.

## What you need, and what you do not

Need: the [README](README.md) topology diagram, and the idea that computers
exchange bytes.

Do not need: binary arithmetic (a `/24` will be explained as "first three
numbers"), the OSI seven-layer poster, or any earlier week of the parent
course.

## 1. The problem networking solves

Two programs want to exchange bytes. They might be on the same computer
(then a Unix socket or `localhost` is enough) or on different computers
(then a network is required).

A network is not a pipe with a single "speed." It is a series of decisions:

1. Which **process** on this computer should receive these bytes?
2. Which **computer** is that process on?
3. Which **neighbor** should this computer hand the bits to next?

Those three questions are why there are three kinds of address. Mixing them
up is the most common source of confusion in the whole track.

## 2. Three addresses, three jobs

| Address | Example in this lab | Job | Visible to |
|---------|---------------------|-----|------------|
| Port | `8080` | Choose a process on a host | The two endpoints |
| IP | `10.20.2.10` | Choose a host, across routers | Every hop, unless NAT later rewrites it |
| MAC | the burned-in (or virtual) NIC address | Choose the next machine on **this cable** | Only that cable; the next hop uses a new pair of MACs |

A useful sentence to memorize:

> IP tells the packet **where it is going**. MAC tells the frame **who
> should take it on this link**. The port tells the destination host
> **which program** should read it.

The client does **not** need the server's MAC. The client needs the MAC of
its gateway (`10.20.1.1` on `r-left`). The router will later use a
**different** MAC pair on the right-hand cable. IP addresses stay the same
on that hop; MAC addresses do not.

### Possible gap: IPv6

IPv6 uses 128-bit addresses (`fd00::10`) and NDP instead of ARP. The three
jobs above are unchanged. This track stays on IPv4 so every example uses
the same four numbers. When you meet `-6` flags later in life, attach them
to this same model.

## 3. IPv4 addresses and prefixes

An IPv4 address is four decimal numbers, each 0–255, written
`10.20.1.10`. Alone, that is just a name. It becomes useful when paired
with a **prefix length**:

```text
10.20.1.10/24
```

The `/24` means: the first 24 bits are the **network**; the remaining 8
bits are the **host**. You do not need to convert to binary to use this.
24 bits is three of the four decimal numbers, so:

- Network (neighborhood): `10.20.1.x`
- This host: `10.20.1.10`
- Other hosts in the same neighborhood: `10.20.1.1`, `10.20.1.2`, …
- A different neighborhood: `10.20.2.10`

Two addresses can talk **directly** (ARP, one cable) only if they are in
the same subnet **and** actually share a link. `10.20.1.10/24` and
`10.20.1.1/24` match both conditions on the left cable.
`10.20.1.10` and `10.20.2.10` do not: different `/24`, so the client
must send the packet to a **router**.

Common prefix lengths you will see:

| Prefix | Meaning in practice |
|--------|---------------------|
| `/32` | Exactly one address. A host route. |
| `/24` | 256 addresses, 254 usable hosts. This lab's subnets. |
| `/16` | 65 536 addresses. A larger neighborhood. |
| `/0` | All addresses. The **default route**. |

In `10.20.1.0/24`:

- `10.20.1.0` is the subnet name, not a host.
- `10.20.1.255` is the broadcast address, not a host.
- `10.20.1.1` is conventionally the gateway; this lab follows that.
- `10.20.1.10` is the client.

Private ranges (RFC 1918) such as `10.0.0.0/8` are ordinary IPv4. They are
just reserved so they are not used as public Internet addresses. The NAT
module is where "private versus public" becomes a forwarding problem.

### Possible gap: you do not need to subnet by hand

Production engineers split networks into `/25`, `/26`, and so on. This
track never asks you to compute that. If a later job requires it, the
skill is "how many host bits remain," not a new theory of routing.

## 4. Hosts, routers, and this lab's two subnets

A **host** originates or terminates traffic. A **router** forwards traffic
that is not for itself.

```text
clp-client              clp-router                 clp-server
10.20.1.10/24     10.20.1.1 | 10.20.2.1      10.20.2.10/24
    c-eth0 -------- r-left   |   r-right -------- s-eth0
         subnet 10.20.1.0/24 | subnet 10.20.2.0/24
```

The router is a host **on both subnets at once**. That is the whole trick.
It has two interfaces, two addresses, and a kernel setting
`net.ipv4.ip_forward=1` that means "if a packet is not for me, try to send
it onward."

Each endpoint also needs a **default route**:

- Client: "anything not in `10.20.1.0/24` goes via `10.20.1.1`."
- Server: "anything not in `10.20.2.0/24` goes via `10.20.2.1`."

A successful request needs **both** directions. If the client can send to
the server but the server has no route back, you will see SYNs arrive and
replies fail. Incident 1 is this failure. Remember the sentence:

> Forward path and return path are two lookups, on two hosts.

## 5. Ethernet, MAC addresses, and why IP is not enough

On a single cable, bits are delivered using **Ethernet frames**. Each
frame has:

- a destination MAC ("who on this cable should wake up");
- a source MAC ("who sent this");
- a type (`0x0800` means "the payload is IPv4"; `0x0806` means ARP);
- a payload (the IP packet, or an ARP message).

The kernel on `clp-client` cannot put the server's MAC in the frame. The
server is not on that cable. It puts the **gateway's** MAC in the
destination, and the **server's IP** in the IP header.

When the router forwards, it **strips the old frame and builds a new
one**. New source MAC: `r-right`. New destination MAC: the server. Same
IP source and destination, TTL decreased by 1.

That is why `tcpdump -e` on `r-left` and `r-right` shows different MACs
and a smaller TTL, but the same IPs (until NAT). If you do not look at
MACs, you can still debug IP problems; if a neighbor is wrong, MACs are
the evidence.

### Possible gap: switches

A real office network has Ethernet switches between hosts. A switch
forwards frames by MAC and does not decrease TTL. This lab's veth cable
is a direct link, which is enough to learn hosts and routers. VLANs,
bridges, and bonding are later topics.

## 6. ARP: asking who has this IP on this cable

The kernel knows the next-hop **IP** from the route table. Ethernet needs
a **MAC**. **ARP** (Address Resolution Protocol) fills the gap:

1. Host broadcasts: "who has `10.20.1.1`? Tell `10.20.1.10`."
2. Owner of that IP replies: "I do; my MAC is …"
3. The answer is cached as a **neighbor** entry.

ARP happens per link. The client ARPs for `10.20.1.1`, never for
`10.20.2.10`. The router ARPs for `10.20.2.10` on the right cable.

Failed neighbors (`nud failed` in `ip neigh`) mean: we needed a MAC and
did not get one. The interface might be down, the IP might not exist on
that link, or ARP replies might be filtered.

IPv6 does the same job with Neighbor Discovery, not ARP. Same idea.

## 7. Encapsulation: one ping across the lab

Suppose the client runs `ping 10.20.2.10`. Ignore DNS; we typed an IP.

**On the client**

1. `ping` asks the kernel to send an ICMP echo request to `10.20.2.10`.
2. Routing: `10.20.2.10` is not in `10.20.1.0/24`. Default route says:
   out `c-eth0`, next hop `10.20.1.1`.
3. Neighbor: if `10.20.1.1` has no MAC yet, ARP first.
4. Frame on the left cable:
   - Ethernet: src = client MAC, dst = `r-left` MAC
   - IP: src = `10.20.1.10`, dst = `10.20.2.10`, TTL typically 64
   - ICMP echo request

**On the router**

1. Frame arrives on `r-left`. Destination MAC matches `r-left`, so the
   kernel unwraps the IP packet.
2. Destination IP is not one of the router's addresses, so this is
   **forward** traffic, not **input** to a local process.
3. If `ip_forward` is 0, the packet dies here. If a firewall's `FORWARD`
   hook drops it, it dies here.
4. Routing: `10.20.2.10` matches `10.20.2.0/24` on `r-right`, on-link.
5. Neighbor: ARP for `10.20.2.10` on the right cable if needed.
6. New frame on the right cable:
   - Ethernet: src = `r-right` MAC, dst = server MAC
   - IP: same addresses, TTL now 63
   - ICMP unchanged

**On the server**

1. Destination IP matches `s-eth0`. This is **input** to a local process
   (the kernel ICMP handler).
2. Echo reply: src `10.20.2.10`, dst `10.20.1.10`.
3. The server looks up **its** route to `10.20.1.10` — a different
   question than the client's lookup. Default via `10.20.2.1`.
4. The reply walks the reverse path: server → `r-right` → `r-left` →
   client. MACs rewrite again; IPs swap roles as source/destination.

If any step fails, ping fails. The rest of the track is how to name which
step.

## 8. Ports, processes, and sockets

ICMP ping never uses a port. TCP and UDP do.

A **socket** is the kernel object a process uses to send and receive. For
TCP it is identified by a **four-tuple**:

```text
source IP : source port  →  destination IP : destination port
```

The **listening** server binds one local address and port, for example
`0.0.0.0:8080`, and waits. `0.0.0.0` means "all IPv4 addresses on this
host." `127.0.0.1:8080` would mean "only processes on this same host can
connect," which is a classic "it works on the server, fails from the
client" bug.

The client picks an **ephemeral** (temporary) source port, such as
`45678`. The four-tuple on the client and the four-tuple on the server
describe the same conversation from opposite ends.

`ss` lists sockets. It cannot see a packet on the wire. `tcpdump` sees the
wire. You will need both.

## 9. TCP, UDP, and ICMP

Three protocols share IP and then behave differently.

**ICMP** reports network conditions and implements ping. Useful as a
reachability probe. Dangerous as a conclusion: firewalls often allow ICMP
and drop TCP, or the reverse.

**UDP** sends datagrams with no handshake. Fast, no connection state in
the protocol. iperf3 can use it to measure loss and jitter. DNS and QUIC
use it on the real Internet; this lab does not require those.

**TCP** builds a reliable byte stream. Before data, the peers complete a
**three-way handshake**:

```text
client → server  SYN      "I want to connect; here is my sequence number"
server → client  SYN-ACK  "I agree; here is mine, and I ack yours"
client → server  ACK      "I ack yours"
```

Only then is the connection `ESTABLISHED`. A listener that is missing
produces **RST** (reset): "nothing here," which `curl` reports as
connection refused. A silent drop produces **timeout**: SYNs repeat, no
RST.

TCP also has sequence numbers, acknowledgements, a receive window, and a
congestion window. You will glance at those with `ss -ti` and iperf3. You
do not need the full congestion-control textbook to finish this track.

### Possible gap: HTTP, TLS, DNS

`curl http://10.20.2.10:8080/` in this lab is TCP to an IP and port, then
a tiny HTTP request. There is no hostname lookup and no encryption. Real
browsers add DNS and TLS **before** they look like this lab's curl. If a
production site "does not load," still prove TCP to the right IP and port
before blaming certificates.

## 10. What can fail: the map you will reuse

For a client request, inspect in this order:

```text
name → socket → local route → neighbor → interface
    → firewall / conntrack → path → remote interface
    → remote route → remote firewall → remote socket
```

The reply must traverse the reverse path. "The request arrived" is half a
successful exchange.

This track has no names (no DNS), so you start at **socket** and **route**.
Firewalls appear in modules 6–8. Performance of a working path is module 5.

At each host or router ask:

1. What source and destination should the packet have here?
2. Which route and outgoing interface should the kernel choose?
3. Is the next-hop neighbor reachable?
4. Does a firewall hook accept it?
5. Does the packet enter and leave as predicted?
6. Is a process listening at the destination?
7. Can the reply return?

## 11. How Linux pretends to be three computers

A Docker container is already a Linux system. Inside it, a **network
namespace** is a smaller box that has its own:

- interfaces;
- addresses;
- routing table;
- ARP cache;
- firewall rules;
- sockets.

It shares the kernel, the filesystem (in this lab), and the CPU with the
other namespaces. That is enough to look like three hosts, and cheap
enough to rebuild in a second.

A **veth pair** is a virtual cable. One end is moved into `clp-client` and
named `c-eth0`. The other end is moved into `clp-router` and named
`r-left`. A frame transmitted at one end arrives at the other.

You almost never type those construction commands in daily work, but you
must understand that `clp-client` is not "a container" and not "a process."
It is a namespace. Commands run **inside** it with:

```bash
ip netns exec clp-client COMMAND
```

Some `ip` commands also take `-n clp-client` as a shortcut. Other tools
(`ss`, `tcpdump`, `ping`, `nft`) need `ip netns exec`.

The container's **root** namespace is not a lab host. Do not flush
`iptables` or `nft` there. Lab scripts only touch names starting with
`clp-`.

### Possible gap: containers, pods, VMs

Kubernetes pods and many containers **are** network namespaces plus extra
policy. A VM has a full kernel. The debugging questions do not change:
addresses, routes, sockets, capture point, firewall. The wrappers around
those objects change. Learn the objects first.

## 12. What this track covers, and what it skips

Covers, in order: the model above; a debugging ritual; then one tool per
module that inspects one part of the model; then incidents that hide a
single fault.

Deliberately skips until you meet them elsewhere:

| Topic | Why it is skipped | What to do if you need it |
|-------|-------------------|---------------------------|
| DNS | The lab uses raw IPs so failures stay in L3/L4 | After this track, `dig` and `/etc/resolv.conf` |
| IPv6 | Same jobs, different spelling | Repeat a module with `-6` on a dual-stack lab |
| VLANs / bridges | Extra L2 wrapping | Week 11 of the parent course, or `ip link type bridge` |
| Dynamic routing (OSPF, BGP) | This lab has one static default route | You already understand the table those protocols fill |
| TLS / HTTPS | Application on top of TCP | Prove TCP first, then `openssl s_client` |
| Cloud SG / NACL / kube-proxy | Different APIs, same hooks | Module 9 lists the translation |

If you feel lost, you are usually missing one of: prefix vs host, host vs
router, forward vs return, or observation point. Return here before adding
tools.

## First look in the lab

Start the environment from the README, then inside the container:

```bash
./scripts/setup-routed.sh
```

Predict each answer, then look:

```bash
ip netns list
ip -n clp-client -br addr
ip -n clp-router -br addr
ip -n clp-server -br addr
ip -n clp-client route
ip -n clp-server route
ip netns exec clp-client ping -c 2 10.20.2.10
```

You should be able to say, without notes:

- which addresses share a subnet;
- why the client has a default route;
- why the router has two addresses;
- which command you used to "sit on" the client.

Reset:

```bash
./scripts/reset.sh
```

## Retrieval test

Without notes, answer in writing:

1. Why can't `clp-client` put `clp-server`'s MAC on its outgoing frame?
2. What does `/24` mean for `10.20.1.10/24`?
3. Why does a ping need a route on the **server**, not only on the client?
4. What is the difference between a port and an IP address?
5. What does `ip netns exec clp-router …` change about a command?
6. Name one thing this track will not teach, and why that is acceptable.

Next: [the debugging method](01-debugging-method.md).
