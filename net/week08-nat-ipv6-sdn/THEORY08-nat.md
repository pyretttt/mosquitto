# Week 08 theory: NAT on Linux

Background reading for the parts of [README08.md](README08.md) that Kurose & Ross 4.5 does not
cover. K&R tells you *that* NAT rewrites addresses and keeps a translation table. It does not tell
you where in the kernel that happens, what the table actually contains, or why the whole thing is
architecturally controversial. That is what this file is for.

This file contains **no answers** to the exercises. Each section ends with a pointer to the
question it arms you for.

Examples use the topology already built by
[../lab/netns/nat-gateway.sh](../lab/netns/nat-gateway.sh), so you can run every command as you
read:

```
  h1                        r                          public
  10.0.0.1/24 ----- 10.0.0.254/24 | 198.51.100.254/24 ----- 198.51.100.1/24
              veth-h1     veth-rin  veth-rout            veth-pub
                          <---- masquerades ---->
```

## Where to look for what

One of three files for this week: **NAT** (this one), [IPv6](THEORY08-ipv6.md),
[SDN](THEORY08-sdn.md).

| README08 item | Section |
| --- | --- |
| Learning goal: SNAT/DNAT/MASQUERADE and what conntrack stores | [Where NAT happens](#where-nat-happens), [SNAT vs DNAT vs MASQUERADE](#snat-vs-dnat-vs-masquerade), [What conntrack actually stores](#what-conntrack-actually-stores) |
| Learning goal: why two hosts behind one NAT can't talk via the public IP | [Hairpin NAT](#hairpin-nat) |
| Lab 1-3: bring up the gateway, watch conntrack, capture post-NAT | [What conntrack actually stores](#what-conntrack-actually-stores) |
| Lab 4: hairpin NAT | [Hairpin NAT](#hairpin-nat) |
| Exercise 1: why does NAT break end-to-end connectivity | [NAT and the end-to-end principle](#nat-and-the-end-to-end-principle) |
| Exercise 1: full cone vs restricted cone vs symmetric | [The cone taxonomy](#the-cone-taxonomy-and-what-replaced-it) |
| Exercise 2: reproduce and fix hairpin NAT | [Hairpin NAT](#hairpin-nat) |
| Exercise 3: nftables vs iptables MASQUERADE | [SNAT vs DNAT vs MASQUERADE](#snat-vs-dnat-vs-masquerade) |
| Self-check: what MASQUERADE does at packet time, what conntrack keeps | [SNAT vs DNAT vs MASQUERADE](#snat-vs-dnat-vs-masquerade), [What conntrack actually stores](#what-conntrack-actually-stores) |
| Self-check: AWS NAT Gateway vs `iptables -j MASQUERADE` | [AWS NAT Gateway vs a Linux box](#aws-nat-gateway-vs-a-linux-box) |

The remaining README08 items — exercise 1's third bullet on `fe80::/10`, lab step 5, the EUI-64
self-check, and everything SDN — are covered in the other two files.

---

## Where NAT happens

Netfilter is a set of five **hooks** in the kernel's IP stack. Anything that wants to see packets
(conntrack, filtering, NAT, logging) registers a callback on a hook with a **priority number**;
lower numbers run first. There is no "iptables engine" separate from this — `iptables` and `nft`
are two front-ends that both compile down to callbacks on the same hooks.

For a packet the router *forwards* (the lab case), the path is:

```
  wire
   |
   v
 [ingress]
   |
   v
 PREROUTING ....... defrag                       (prio -400)
   |                conntrack: lookup or create  (prio -200)
   |                nat: DNAT                    (prio -100)
   v
 routing decision   <-- reads the destination address, post-DNAT
   |
   v
 FORWARD .......... filter                       (prio    0)
   |
   v
 POSTROUTING ...... nat: SNAT / MASQUERADE       (prio +100)
   |                conntrack: confirm           (prio INT_MAX)
   v
 [egress]
   |
   v
  wire
```

Packets destined for the router itself go `PREROUTING -> routing -> INPUT -> process`, and packets
the router originates go `process -> OUTPUT -> routing -> POSTROUTING`. NAT is available on
`OUTPUT` (DNAT, prio -100) and `INPUT` (SNAT, prio +100) too, which is how you redirect traffic a
box generates for itself.

Two facts fall out of those priority numbers, and they are the whole reason the hook order matters:

- **DNAT must run before the routing decision.** DNAT changes the destination address. The router
  has not chosen an output interface yet, and it must choose it based on the *new* destination.
  Run DNAT after routing and you would route to the old destination and then rewrite the header —
  the packet would leave the wrong interface.
- **SNAT must run after the routing decision.** For `MASQUERADE` the source address to use *is*
  the address of the interface routing picked. You cannot know it before routing runs. Even for
  plain SNAT, source rewriting must not perturb the routing lookup, and delaying it until
  POSTROUTING guarantees that.

Verify the priorities yourself:

```bash
nft list ruleset            # look at the "priority" keyword on each chain
nft describe priority       # nft's symbolic names: raw, mangle, dstnat, filter, srcnat
```

The symbolic names `dstnat` (-100) and `srcnat` (+100) are what the numbers in
`nat-gateway.sh` (`priority 100`) refer to.

> **Only the first packet of a flow is evaluated against the nat chains.** The nat hook is only
> consulted for packets conntrack marks as `NEW` (plus `RELATED`, for helper-created expectations).
> Once a flow has a NAT binding attached to its conntrack entry, every subsequent packet is
> translated straight from the binding without re-running a single rule. This is why NAT rulesets
> can be arbitrarily expensive to evaluate without costing you throughput, and it is also why
> changing a NAT rule does not affect flows already established.

*Arms you for:* the self-check on what MASQUERADE does at packet time.

---

## SNAT vs DNAT vs MASQUERADE

All three are **stateful NAPT** (Network Address *and Port* Translation, RFC 2663's "NAPT" as
opposed to "basic NAT" which is address-only 1:1). They differ in which half of the tuple they
rewrite and where the replacement address comes from.

| | SNAT | DNAT | MASQUERADE |
| --- | --- | --- | --- |
| Rewrites | source addr (+ port) | destination addr (+ port) | source addr (+ port) |
| Valid hooks | postrouting, input | prerouting, output | postrouting only |
| New address comes from | the rule (`--to-source`) | the rule (`--to-destination`) | the egress interface, read at packet time |
| Cost per new flow | none beyond the rewrite | none beyond the rewrite | one interface-address lookup |
| Survives the interface changing IP | no, rule is now wrong | n/a | yes |
| On interface down | entries persist | entries persist | its conntrack entries are flushed |
| Typical use | fixed public IP, egress from a known range | port forwarding, load balancing | DHCP/PPPoE uplink, container egress |

### What MASQUERADE actually is

`MASQUERADE` is not a separate mechanism. It is SNAT with the source address left as a runtime
lookup instead of a rule argument. At packet time, for the first packet of a flow, it:

1. Reads the output device the routing decision selected.
2. Takes the primary IPv4 address of that device (`ipv4_devconf` / the first address in the
   device's address list matching the route's scope).
3. Performs SNAT to that address, allocating a source port.
4. Records the resulting binding on the conntrack entry.

Two consequences you should be able to state without notes. First, it works on a link whose
address is assigned by DHCP or PPP and changes underneath you — no rule edit needed. Second,
because a mapping to an address that no longer exists is worse than no mapping, the masquerade
module registers for device notifications and **flushes the conntrack entries bound to that device
when it goes down**. Plain SNAT does not do this, which is why a static SNAT rule plus a changing
uplink IP produces the classic "connections hang forever after the PPP session flapped" bug.

The price is a per-flow address lookup that plain SNAT does not pay. On a box with a genuinely
static public address, `-j SNAT --to-source x.x.x.x` is marginally cheaper. On anything else, use
`MASQUERADE`.

### The same rules in both front-ends

Egress masquerade, matching what `nat-gateway.sh` installs:

```bash
# iptables
iptables -t nat -A POSTROUTING -o veth-rout -j MASQUERADE

# nftables
nft add table ip nat
nft 'add chain ip nat postrouting { type nat hook postrouting priority srcnat ; policy accept ; }'
nft add rule ip nat postrouting oifname "veth-rout" masquerade
```

Static SNAT for a known source range:

```bash
iptables -t nat -A POSTROUTING -s 10.0.0.0/24 -o veth-rout -j SNAT --to-source 198.51.100.254
nft add rule ip nat postrouting ip saddr 10.0.0.0/24 oifname "veth-rout" snat to 198.51.100.254
```

Port forward (DNAT) of the public 8080 to h1:

```bash
iptables -t nat -A PREROUTING -i veth-rout -p tcp --dport 8080 -j DNAT --to-destination 10.0.0.1:8080
nft add rule ip nat prerouting iifname "veth-rout" tcp dport 8080 dnat to 10.0.0.1:8080
```

For exercise 3, the interesting differences when you list them back:

```bash
iptables -t nat -L -n -v      # per-chain listing, counters inline, tables are fixed and implicit
iptables-save -t nat          # the form actually worth diffing
nft list table ip nat         # one nested document: table -> chain -> rule, with the hook and priority visible
nft -a list table ip nat      # adds handles, which is how you delete a specific rule
```

Things to note in the write-up: `iptables` has five fixed built-in tables whose relative order you
have to memorise, whereas `nft` makes you declare the hook and priority explicitly so the ordering
is visible in the ruleset itself; `nft` has one `inet` family that covers IPv4 and IPv6 in a single
ruleset (which is why the lab script writes `table inet nat`, not `table ip nat`); and `nft` sets
no default chains, so a table with no chains matches nothing rather than falling through to a
policy.

*Arms you for:* exercise 3, and the self-check on what MASQUERADE does at packet time.

---

## What conntrack actually stores

K&R draws the NAT translation table as one row: `(private ip, private port) <-> (public ip, public
port)`. The real structure is different and better, and understanding it makes the return-path
rewrite obvious rather than magic.

### Two tuples, not one row

A conntrack **tuple** (`struct nf_conntrack_tuple`) is everything needed to identify one direction
of a flow: source address, destination address, L3 protocol, L4 protocol, and the protocol's
identifying pair — source port and destination port for TCP/UDP, or the ICMP id and type.

Each connection (`struct nf_conn`) holds **two** tuples:

- `IP_CT_DIR_ORIGINAL` — the tuple of the first packet, as it arrived, before translation.
- `IP_CT_DIR_REPLY` — the tuple the kernel **expects to see on the wire** for the return traffic.

With no NAT in play, the reply tuple is just the original inverted. **NAT is implemented by
writing a reply tuple that is not the inverse.** There is no separate NAT table; the translation
*is* the difference between the two tuples.

### Reading real output

Run the lab, then:

```bash
sudo ip netns exec public nc -lvnp 8080 &
sudo ip netns exec h1 bash -c 'echo hello | nc 198.51.100.1 8080'
sudo ip netns exec r conntrack -L
```

```
tcp  6  431999  ESTABLISHED  src=10.0.0.1       dst=198.51.100.1    sport=51234  dport=8080
                             src=198.51.100.1   dst=198.51.100.254  sport=8080   dport=51234
                             [ASSURED] mark=0 use=1
```

Field by field:

- `tcp 6` — L4 protocol name and number.
- `431999` — seconds until this entry is garbage-collected if it sees no more packets. 432000 is
  the default for an established TCP connection (five days, `nf_conntrack_tcp_timeout_established`).
- `ESTABLISHED` — the **TCP** state machine's state, tracked separately from conntrack's own
  NEW/ESTABLISHED/RELATED states that you match on in rules. Do not conflate them: a UDP flow can
  be conntrack-ESTABLISHED and has no TCP state at all.
- First `src=/dst=/sport=/dport=` group — the original tuple. h1's real address and port.
- Second group — the reply tuple.

Now the thing to actually stare at. The reply tuple's `dst` is `198.51.100.254`, **not**
`10.0.0.1`. It is not the mirror image of the original tuple. It is a literal description of the
packet the router expects to receive back: "a packet from 198.51.100.1:8080 addressed to
198.51.100.254:51234". That is exactly what will show up on `veth-rout`, because that is what the
public host is replying to.

So the return path needs no cleverness:

1. Return packet arrives. Conntrack builds its tuple: `src=198.51.100.1 dst=198.51.100.254
   sport=8080 dport=51234`.
2. Hash lookup on that tuple hits this entry, and the match is in the **reply** direction.
3. Because the match was in the reply direction, the NAT engine applies the *inverse* of the
   recorded translation: rewrite the destination to the original tuple's source,
   `10.0.0.1:51234`.
4. Routing sends it out `veth-rin` to h1, which sees a reply to the socket it actually opened.

The reply tuple is simultaneously the lookup key for return traffic and the record of what was
translated. That is the whole design.

Watch it happen live in a second terminal while you generate traffic:

```bash
sudo ip netns exec r conntrack -E        # event stream: NEW / UPDATE / DESTROY per flow
```

### Where the port came from

`sport=51234` is preserved in the example, and that is the normal case: Linux tries hard to keep
the original source port, because port preservation is what makes a Linux NAT look
well-behaved to peer-to-peer applications. It only picks a different port when the tuple it wants
is already taken by another flow. When it has to choose, it allocates within a range derived from
the original port (below 512, 512-1023, or 1024-65535) so a privileged source port stays
privileged.

You can override the policy:

```bash
nft add rule ip nat postrouting oifname "veth-rout" masquerade random        # randomize port
nft add rule ip nat postrouting oifname "veth-rout" masquerade fully-random  # stronger randomization
nft add rule ip nat postrouting oifname "veth-rout" masquerade persistent    # same client -> same public IP across a pool
```

Port preservation matters again in [the cone taxonomy](#the-cone-taxonomy-and-what-replaced-it),
because it is the mechanism that makes Linux behave like a cone NAT rather than a symmetric one.

### Flags, and why entries are not inserted immediately

- `[UNREPLIED]` — nothing has been seen in the reply direction yet. You will see this on the first
  packet, and permanently on flows to a black hole.
- `[ASSURED]` — traffic has been seen in both directions and the flow is considered real. Under
  table pressure the kernel's early-drop logic evicts non-assured entries first, so this flag is
  what protects your working connections from a SYN flood.

An entry is *created* by the conntrack hook at prerouting (priority -200) but is **not inserted
into the global hash table** until the packet survives all the way to the confirm hook (priority
`INT_MAX`, at postrouting or input). In between it lives attached to the `sk_buff`. If a filter
rule drops the packet, the entry is discarded and never appears in `conntrack -L`. This is why a
dropped packet costs you no table slot, and why the "unconfirmed" list exists in the kernel at all.

### When the table fills

```bash
sysctl net.netfilter.nf_conntrack_count      # current
sysctl net.netfilter.nf_conntrack_max        # limit
conntrack -S                                 # per-CPU counters
```

The failure mode is nastier than most: when the table is full, packets that would create a new
flow are **hard-dropped in the conntrack hook**, before any rule you wrote is consulted. There is
no rule to find, no counter in your ruleset that increments, and locally-generated traffic starts
getting `EPERM` back from `sendto()`. In `dmesg` you get `nf_conntrack: table full, dropping
packet`; in `conntrack -S` you get rising `insert_failed` and `drop`.

The same `insert_failed` counter also rises for a distinct NAT-specific reason: the port space for
some `(public ip, protocol, destination)` combination is exhausted and no free tuple can be
allocated. Remember this one — it is the direct analogue of AWS's `ErrorPortAllocation`, discussed
[below](#aws-nat-gateway-vs-a-linux-box).

*Arms you for:* the self-check on what conntrack keeps so the return packet is rewritten correctly.

---

## Hairpin NAT

Also called NAT loopback or NAT reflection. Named for the packet's U-turn: in and back out the
same interface.

### The setup that fails

Extend the lab with `h3 = 10.0.0.3/24` on the inside, run a service on h1 port 8080, and publish
it on the router:

```bash
nft add rule ip nat prerouting tcp dport 8080 dnat to 10.0.0.1:8080
```

From the outside world this works. From h3, `nc 198.51.100.254 8080` hangs or resets. Nothing is
misconfigured in the obvious sense — the port forward is correct, the routes are correct, both
hosts are up.

### Trace it packet by packet

1. **h3 sends.** h3 has no route to `198.51.100.0/24` other than its default route, so the SYN
   goes to the router. On the wire: `src=10.0.0.3:40000 dst=198.51.100.254:8080`.
2. **r DNATs it.** The rule above is not scoped to an input interface, so it matches. In
   PREROUTING the destination becomes `10.0.0.1:8080`. Packet is now
   `src=10.0.0.3:40000 dst=10.0.0.1:8080`.
3. **r routes it back where it came from.** `10.0.0.1` is on `veth-rin` — the same interface the
   packet arrived on. The router forwards it back out the inside interface. It also, by default,
   emits an ICMP redirect to h3 saying "for 10.0.0.1, go direct."
   Nothing has gone wrong *yet*. The SNAT chain in POSTROUTING has no rule matching this packet,
   so the source is left as `10.0.0.3`.
4. **h1 replies — directly.** h1 receives a SYN from `10.0.0.3`. Its route to `10.0.0.3` is the
   directly-connected LAN. So the SYN/ACK goes straight to h3 across the wire:
   `src=10.0.0.1:8080 dst=10.0.0.3:40000`. **It never touches the router.**
5. **h3 rejects it.** h3 has a socket in `SYN_SENT` to `198.51.100.254:8080`. A segment arrives
   from `10.0.0.1:8080`. No socket matches that 4-tuple, so h3 answers with a RST (or its conntrack
   marks it INVALID and drops it). The handshake never completes.

The one-sentence version, which is the version worth remembering: **the router translated the
request but never got the chance to un-translate the reply, because the reply took a shortcut that
bypassed the router entirely.** Half a NAT is worse than none.

A variant worth knowing: if you had scoped the DNAT rule to the outside interface
(`iifname "veth-rout" tcp dport 8080 dnat to ...`), step 2 never happens. The packet is instead
delivered to the router itself, which has nothing listening on 8080, and h3 gets an immediate
connection refused. Same underlying cause, different symptom — which is why "it works from my
phone on 4G but not from my laptop on the LAN" is the canonical bug report for this.

### Reproducing it cleanly

```bash
# terminal 1 - watch the inside wire
sudo ip netns exec h3 tcpdump -ni veth-h3 -e

# terminal 2 - is the router even seeing the return traffic?
sudo ip netns exec r conntrack -E

# terminal 3
sudo ip netns exec h1 bash -c 'while true; do nc -l -p 8080; done' &
sudo ip netns exec h3 nc -v 198.51.100.254 8080
```

The diagnostic signature: an `[UNREPLIED]` conntrack entry on r that never becomes `[ASSURED]`,
plus a SYN/ACK visible in h3's capture whose source address is not the address h3 dialled. Save
that capture — it is the "before" artifact for exercise 2.

### The fix, and its cost

Force the reply back through the router by also SNATing the hairpinned leg, so h1 believes it is
talking to the router rather than to h3:

```bash
# nftables
nft add rule ip nat postrouting ip saddr 10.0.0.0/24 ip daddr 10.0.0.1 tcp dport 8080 \
    snat to 10.0.0.254

# iptables
iptables -t nat -A POSTROUTING -s 10.0.0.0/24 -d 10.0.0.1 -p tcp --dport 8080 \
    -j SNAT --to-source 10.0.0.254
```

The match condition is the general form of "this flow entered from the inside and is leaving to the
inside" — source in the local prefix, destination the DNAT target. Now the flow is
`10.0.0.3 -> 198.51.100.254` becomes `10.0.0.254 -> 10.0.0.1`, h1 replies to `10.0.0.254`, the
router matches the reply tuple, undoes both translations in one pass, and h3 sees a reply from the
address it dialled.

The cost is not subtle: **h1 now sees every hairpinned client as `10.0.0.254`**. Access logs,
per-client rate limits, and source-IP ACLs on h1 are all blinded for internal clients. This is the
same trade that makes `X-Forwarded-For` necessary in front of L7 proxies, and the same reason
Kubernetes services default to `externalTrafficPolicy: Cluster` losing the client IP. Note it in
your write-up; it is the interesting half of the answer.

RFC 4787 REQ-9 makes hairpinning mandatory for a compliant NAT precisely because so many devices
got this wrong.

> **Name collision worth knowing.** `bridge link set dev X hairpin on` and kubelet's `hairpinMode`
> are a *layer-2* feature: permitting a bridge to forward a frame back out the port it arrived on,
> which is normally forbidden. Related in spirit, different layer, different fix. If a pod cannot
> reach itself through its own service IP in Kubernetes, you may be looking at either one.

*Arms you for:* the learning goal on hairpin NAT, and exercise 2.

---

## NAT and the end-to-end principle

The original IP architecture made a specific promise: **every host has a globally unique address,
any host can send a packet to any other, and all per-connection state lives in the endpoints.**
Routers were supposed to be stateless forwarders that could be swapped out mid-conversation
without anyone noticing. NAT violates all three parts. They fail in different ways, so keep them
separate when you write this up.

### 1. Addresses stop being unique, so they stop being names

`10.0.0.1` identifies a different machine in every network on earth. The consequence is not just
philosophical — it breaks **referral**. Under the original model you could hand your address to a
third party and they could reach you. Behind NAT, an address is only meaningful relative to a
namespace you have no way to name. Every protocol that passes an address around inside its payload
(FTP's `PORT`, SIP's SDP body, older RPC) inherits this bug.

### 2. Inbound reachability is gone, and hosts split into two classes

A NAT with no matching state has no way to know where an unsolicited inbound packet should go, so
it drops it. Communication becomes possible only if an *inside* host speaks first. The internet's
symmetric peer model collapses into client and server, and being a server now requires
configuration on a box you may not control. Note that this is a *policy* decision — who may talk
to whom — that has been welded into a forwarding device as a side effect of an addressing hack.
The security-by-accident is real, but it is a byproduct, not a design.

### 3. Fate sharing is broken

David Clark's "fate sharing" principle from the DARPA design papers: state about a conversation
should be stored only where its loss is equivalent to the loss of the conversation, i.e. in the
endpoints. A NAT holds per-flow state in the middle. Reboot it and every flow dies, even though
both endpoints are perfectly healthy and their sockets still think they are connected. You cannot
transparently fail over to a second NAT box without replicating that state — which is exactly why
`conntrackd` exists, and why a stateful middlebox is always the hardest part of an HA design.
Compare: you can renumber and reboot a *router* mid-flow and nothing breaks.

### 4. Layer violations, and the incompatibility with integrity protection

A NAT cannot operate on the IP header alone:

- It must rewrite L4 ports, which means recomputing the TCP/UDP checksum — and those checksums
  cover a pseudo-header containing the IP addresses. A device that was supposed to be L3 is now
  required to parse and mutate L4.
- ICMP errors quote the original IP header in their payload. To pass an ICMP "fragmentation
  needed" or "port unreachable" back to the right inside host, the NAT must reach into the ICMP
  body and rewrite the *embedded* header too. Get this wrong and you get PMTU black holes (see
  week 07).
- Protocols carrying addresses in their payload need L7 awareness: `nf_conntrack_ftp`,
  `nf_conntrack_sip` and friends, the "connection tracking helpers" / ALGs. They are a security
  liability and are disabled by default in modern kernels.
- **IPsec AH authenticates the IP header, including the addresses NAT rewrites.** The two
  mechanisms are structurally incompatible: NAT invalidates the ICV by definition. The industry
  response was NAT-T, which wraps ESP in UDP/4500 — tunnelling around the middlebox rather than
  fixing it.

### 5. The knock-on: transport ossification

Because NATs only understand TCP, UDP and ICMP, a new transport protocol is undeployable on the
public internet. SCTP and DCCP are the casualties. This is a large part of why QUIC is built on
UDP rather than being a new IP protocol number: not because UDP is a good substrate, but because
it is the only one that traverses the installed base of middleboxes.

### The counter-argument, which you should read before writing

Geoff Huston's ["In defence of NATs"](https://blog.apnic.net/2017/09/06/opinion-defence-nats/)
argues the conventional conclusion is wrong: NAT reframes the IP address from a durable host
identity into an ephemeral per-session token, which is arguably a *better* fit for how the network
is actually used, and delivers a privacy property the original architecture never had. Whether or
not you buy it, a write-up that only recites the complaints is a weaker write-up than one that
takes the strongest opposing case seriously.

*Arms you for:* exercise 1, first bullet.

---

## The cone taxonomy, and what replaced it

Set up one concrete scenario and hold it fixed; every definition below is a statement about this
scenario.

```
inside host X = 10.0.0.1:5000
NAT public    = 198.51.100.254
server S1     = 203.0.113.1:9999
server S2     = 203.0.113.2:9999
peer P        = 192.0.2.7:1234

X sends to S1. The NAT creates the mapping:
    10.0.0.1:5000  <->  198.51.100.254:40000
```

There are two entirely independent questions to ask about the NAT, and the classic taxonomy's flaw
is that it tangles them together.

**Question A — mapping behavior.** X now sends to `S2`, a different destination, from the same
source port 5000. Does the NAT reuse external port 40000, or allocate a new one?

**Question B — filtering behavior.** Who is permitted to send an inbound packet to
`198.51.100.254:40000` and have it delivered to X?

### The four classic types (RFC 3489)

| Type | Reuses :40000 for S2? | Inbound to :40000 accepted from |
| --- | --- | --- |
| Full cone | yes | anyone, including P, unsolicited |
| Restricted cone | yes | any port on 203.0.113.1 |
| Port-restricted cone | yes | only 203.0.113.1:9999 |
| Symmetric | **no** — new external port per destination | only 203.0.113.1:9999 |

"Cone" is a picture: one internal endpoint fans out to many external peers through a *single*
external port, so the mapping looks like a cone with its apex at the inside host. A symmetric NAT
has no apex — each destination gets its own external port, so the picture is a bundle of separate
one-to-one pipes.

The three cone types differ only in Question B; they answer Question A identically. Symmetric is
the only one that answers Question A differently. That asymmetry in the taxonomy is a hint that it
is carved wrong.

### RFC 4787's two axes

RFC 4787 deprecated the cone vocabulary and split it into the two orthogonal properties:

**Mapping behavior** — what the external port depends on:

- *Endpoint-Independent Mapping* (EIM): depends only on the internal source. One port for all
  destinations.
- *Address-Dependent Mapping*: also depends on the destination address.
- *Address-and-Port-Dependent Mapping*: also depends on the destination address and port.

**Filtering behavior** — what inbound traffic is accepted:

- *Endpoint-Independent Filtering*: anyone.
- *Address-Dependent Filtering*: only hosts the inside host has sent to.
- *Address-and-Port-Dependent Filtering*: only the exact `ip:port` the inside host sent to.

The translation table back to the old names:

| Classic name | RFC 4787 |
| --- | --- |
| Full cone | EIM + Endpoint-Independent Filtering |
| Restricted cone | EIM + Address-Dependent Filtering |
| Port-restricted cone | EIM + Address-and-Port-Dependent Filtering |
| Symmetric | Address-and-Port-Dependent Mapping (+ the matching filtering) |

**RFC 4787 REQ-1: a NAT MUST have Endpoint-Independent Mapping behavior.** Filtering is left to
policy (REQ-8 recommends endpoint-independent or address-dependent). In other words, the standard
outlawed symmetric NAT and left the choice among the three cone types to the operator — which is
precisely the split the two-axis model makes visible and the four-name model hides.

The taxonomy was deprecated because real devices do not sit in four buckets. Behavior varies by
protocol, changes when the port pool comes under pressure, differs for a refreshed mapping versus
a fresh one, and interacts with port-preservation heuristics. A device can be EIM until it hits a
port collision and then silently degrade. "Full cone" is a claim about one moment in a device's
life, not a property of the device.

### Where Linux sits

Netfilter stores the mapping on the conntrack entry, and — crucially — tries to preserve the
original source port. When X opens a second flow from `10.0.0.1:5000` to `S2`, port 40000 is not
in conflict (the destination differs, so the full tuple differs), so it is reused. Linux therefore
*behaves* as **endpoint-independent mapping with address-and-port-dependent filtering**: a
port-restricted cone NAT. But this is emergent from port preservation rather than a guarantee.
Under port collision, or with `masquerade random`, Linux allocates a fresh port per destination and
degrades to symmetric behavior for that flow. Worth verifying empirically rather than believing —
`stunclient` or two `nc -p 5000` flows to different destinations plus `conntrack -L` will tell you
what your kernel actually did.

### Why symmetric NAT breaks hole punching

This is the payoff, and the reason anyone cares about the taxonomy.

STUN's whole trick is: X asks a STUN server "what source address and port do you see?" The server
answers `198.51.100.254:40000`. X sends that to peer P out of band. P sends to
`198.51.100.254:40000`, X simultaneously sends to P (opening the filter for P's address), and both
sides' packets arrive. This works **only if the port X learned from the STUN server is the same
port X will use towards P** — that is, only under endpoint-independent mapping.

Under symmetric NAT it isn't. X's flow to P gets a different external port, say 40001, and the
address X advertised is dead on arrival. Port prediction (guessing 40001) works against naive
sequential allocators and fails against randomized ones and against carrier-grade NAT. If either
side is symmetric, ICE typically has to fall back to a **TURN relay**: both peers connect outbound
to a public server that forwards between them, which costs latency and bandwidth and reintroduces
the middlebox you were trying to escape.

*Arms you for:* exercise 1, second bullet.

---

## AWS NAT Gateway vs a Linux box

Both do stateful NAPT for egress. The self-check is really asking where the abstraction leaks.

| | `iptables -j MASQUERADE` | AWS NAT Gateway |
| --- | --- | --- |
| Scaling unit | your instance's CPU and conntrack table | managed, scales to 100 Gbps |
| Connection limit | `nf_conntrack_max`, tunable, global | 55,000 concurrent **per IPv4 address per unique destination** (dest IP + dest port + protocol); up to 8 addresses, so ~440,000 |
| Source port range | derived from the original port, preserved where possible | 1024-65535, always reallocated |
| Which source IP | deterministic: primary address of the egress interface | with multiple IPs, chosen by a **flow hash** over ENI id, addresses, ports, protocol, and TCP sequence number |
| Inbound / DNAT | yes, you own the ruleset | none. No port forwarding, ever |
| Security groups | n/a, you write the filter rules | has none of its own; filter with NACLs and the instances' SGs |
| Idle TCP timeout | 432000s established by default, tunable | **350s**, not tunable |
| After idle timeout | entry expires silently, next packet is a new flow | returns RST to whichever side speaks next |
| MTU / MSS | you add `TCPMSS --clamp-mss-to-pmtu` yourself | MTU 8500 enforced, MSS clamping automatic, larger packets dropped |
| Fragments | handled by the defrag hook | forwards fragmented UDP; **drops fragmented TCP and ICMP** |
| Failure telemetry | `conntrack -S` (`insert_failed`, `drop`), `dmesg` | CloudWatch `ErrorPortAllocation`, `PacketsDropCount`, `IdleTimeoutCount` |
| Availability | your box is a SPOF; HA needs `conntrackd` state sync | redundant *within one AZ*; an AZ needs its own gateway |
| Cost | free, you already pay for the instance | per-hour plus per-GB processed |

The three points most worth internalising:

1. **The connection limit is per-destination, not global.** 55,000 sounds tiny until you realise it
   is 55,000 to *each* `(ip, port, protocol)`. It only bites when a fleet hammers a single endpoint
   — an S3 VPC endpoint bypass, one busy API, a shared database. That is also the exact shape of
   the Linux failure: `insert_failed` from port exhaustion for one destination tuple, not from the
   table being globally full. Same bug, two dashboards.
2. **Source IP selection is non-deterministic across a multi-IP gateway.** MASQUERADE gives you one
   predictable egress address per interface; NAT Gateway with several addresses hashes per flow,
   including the TCP sequence number in the hash. Any downstream partner doing source-IP
   allowlisting must allowlist all of them, and you cannot reason about which one a given instance
   will use.
3. **350 seconds.** This is the single most common production surprise. A long-lived idle
   connection — a database pool, a gRPC channel, a queue consumer — dies silently and the
   application discovers it on the next write. Set TCP keepalives below 350s. There is no
   equivalent trap on Linux with its five-day default, which is exactly why people are unprepared
   for it.

Also worth stating for the answer: NAT Gateway performs PAT (many-to-one), not 1:1 NAT, and its
translation happens in two stages — the gateway rewrites to its private address, then the Internet
Gateway performs the second translation to the Elastic IP. And it is not a general-purpose box: no
custom rules, no DNAT, no conntrack introspection. When you need any of those, you are back to a
NAT *instance* running the ruleset from this file, and you own its availability.

*Arms you for:* the self-check on AWS NAT Gateway vs MASQUERADE.

---

## Reading list

All verified reachable. Where a piece is long, the section to actually read is named.

**Hooks, SNAT/DNAT/MASQUERADE**

- [Linux 2.4 NAT HOWTO, ch.6 "Saying How To Mangle The Packets"](https://www.iptables.org/documentation/HOWTO/NAT-HOWTO-6.html)
  — Rusty Russell's original, still the crispest statement of the SNAT/DNAT/MASQUERADE split.
  Read 6.1-6.3; 6.3 "Mappings In Depth" covers implicit source-port remapping.
- [Netfilter hooks (nftables wiki)](https://wiki.nftables.org/wiki-nftables/index.php/Netfilter_hooks)
  — the authoritative priority table. This is the one page that explains the ordering constraint.
- [Performing NAT (nftables wiki)](https://wiki.nftables.org/wiki-nftables/index.php/Performing_Network_Address_Translation_\(NAT\))
  — modern syntax, and states explicitly that only the first packet of a flow triggers a rule
  lookup. Also covers `random` / `fully-random` / `persistent`.
- [Nftables - packet flow and Netfilter hooks in detail (Thermalcircle)](https://thermalcircle.de/doku.php?id=blog:linux:nftables_packet_flow_netfilter_hooks_detail)
  — article-length walkthrough of Jan Engelhardt's diagram, including per-family hooks and
  netns behavior.
- [File:Netfilter-packet-flow.svg](https://en.wikipedia.org/wiki/File:Netfilter-packet-flow.svg)
  — the canonical diagram itself. Open the full-resolution SVG; it is illegible at thumbnail size.

**conntrack**

- [NAT part 2: the conntrack tool (Fedora Magazine)](https://fedoramagazine.org/network-address-translation-part-2-the-conntrack-tool/)
  — by netfilter maintainer Florian Westphal. Start here. This is the best short explanation of the
  two-tuple model anywhere, with real `conntrack -L` output.
- [NAT part 1: packet tracing](https://fedoramagazine.org/network-address-translation-part-1-packet-tracing/)
  — `nft monitor trace` with a full worked debugging session on a broken port-forward. Do this
  during the lab.
- [NAT part 4: conntrack troubleshooting](https://fedoramagazine.org/network-address-translation-part-4-conntrack-troubleshooting/)
  — per-protocol validity checks, `nf_conntrack_log_invalid`, the unconfirmed and dying lists.
- [Connection tracking: design and implementation inside the Linux kernel (Arthur Chiao)](http://arthurchiao.art/blog/conntrack-design-and-implementation/)
  — the deep dive. `struct nf_conntrack_tuple`, `nf_conntrack_in()`, `nf_conntrack_confirm()`, then
  a separate section on the NAT engine and masquerade. Read the tuple and NAT sections; the rest is
  reference.
- [Connection tracking parts 1-3 (Thermalcircle)](https://thermalcircle.de/doku.php?id=blog%3Alinux%3Aconnection_tracking_1_modules_and_hooks)
  — based on kernel 5.10, so more current than most. Part 2 for timeouts and the hash table,
  part 3 for status bits and NEW/ESTABLISHED/RELATED with worked TCP, UDP and ICMP examples.
- [Conntrack tales: one thousand and one flows (Cloudflare)](https://blog.cloudflare.com/conntrack-tales-one-thousand-and-one-flows/)
  — Marek Majkowski demonstrates table exhaustion experimentally, including the silent hard-drop
  and `EPERM` from `sendto()`. Reproducible with `unshare`.
- [conntrack-tools manual](https://conntrack-tools.netfilter.org/manual.html)
  — CLI reference for `conntrack` and `conntrackd`. Lookup only; not always current, cross-check
  against the man pages.

**Hairpin**

- [Loopback to forwarded public IP from the local network (ServerFault)](https://serverfault.com/questions/55611/loopback-to-forwarded-public-ip-address-from-local-network-hairpin-nat)
  — the canonical explanation, and it gets the causal chain right. Read this before writing
  exercise 2.
- [Configure hairpin on ASA (Cisco)](https://www.cisco.com/c/en/us/support/docs/security/secure-firewall-threat-defense/221949-configure-hairpin-on-asa.html)
  — read the conceptual intro only, for the naming (hairpin / NAT loopback / NAT reflection) and
  confirmation that this is not a Linux quirk.

**Cone types and traversal**

- [How NAT traversal works (Tailscale)](https://tailscale.com/blog/how-nat-traversal-works)
  — the best single explainer in this list, built from first principles. It is long; for exercise 1
  the sections on firewalls, NAT mapping behavior, and STUN are enough. The hole-punching and
  relay sections are worth the rest of the read anyway.
- [Network address translation (Wikipedia)](https://en.wikipedia.org/wiki/Network_address_translation)
  — has the exact cone-name to RFC-4787-name mapping table, and explains why RFC 3489's
  classification was deprecated.
- [NAT traversal (Wikipedia)](https://en.wikipedia.org/wiki/NAT_traversal)
  — hole punching over UDP/TCP/ICMP, why symmetric NAT defeats STUN and ICE, port prediction and
  why it fails against CGN, and IPsec NAT-T.
- [WebRTC for the Curious: Connecting](https://webrtcforthecurious.com/docs/03-connecting/)
  — the same problem from the standardised-protocol side: ICE agents, candidates, pairing, and
  where STUN and TURN sit.

**End-to-end**

- [Opinion: in defence of NATs (Geoff Huston, APNIC)](https://blog.apnic.net/2017/09/06/opinion-defence-nats/)
  — states the objection clearly, then argues against the usual conclusion. Read the opposing case.
  Mirrored at [potaroo.net](http://www.potaroo.net/ispcol/2017-09/natdefence.html).
- [Why are NATs so popular? (Geoff Huston, 2003)](https://www.potaroo.net/ispcol/2003-09/nats.html)
  — the earlier, more critical piece; sharpest on the concrete IPsec AH incompatibility.
- [Rethinking the design of the internet: the end-to-end arguments vs. the brave new world (Blumenthal & Clark)](https://cs.nyu.edu/~lakshmi/classes/networks/clark00.pdf)
  — a paper rather than an article, but it is the foundational statement by one of the original
  end-to-end authors. Sections 1-3 are the relevant part.

**AWS**

- [NAT gateway basics (AWS VPC User Guide)](https://docs.aws.amazon.com/vpc/latest/userguide/nat-gateway-basics.html)
  — primary source for every number in the table above.
- [Attach multiple IPs to a NAT Gateway to scale egress (AWS Networking blog)](https://aws.amazon.com/blogs/networking-and-content-delivery/attach-multiple-ips-to-a-nat-gateway-to-scale-your-egress-traffic-pattern/)
  — the closest thing to an internals writeup: PAT rather than 1:1, two-stage translation via the
  IGW, and the flow-hash source-IP selection.
- [Resolve port allocation errors on a NAT gateway (AWS re:Post)](https://repost.aws/knowledge-center/vpc-resolve-port-allocation-errors)
  — the operational view of `ErrorPortAllocation`, with CloudWatch Insights queries to find the
  destination exhausting your ports.

## RFCs, lookup only

You should not read these end to end. If you need one, here is the part that matters.

| RFC | Look at | For |
| --- | --- | --- |
| 4787 | REQ-1 (§4.1), REQ-8 (§5), REQ-9 (§6) | mapping behavior, filtering behavior, mandatory hairpinning |
| 5128 | §2 | the definitive terminology table for the cone types |
| 5128 | §3.3-3.4 | UDP and TCP hole punching mechanics |
| 2663 | §4 | basic NAT vs NAPT, and the traditional vocabulary |
| 3022 | §2 | traditional/outbound NAT operation |
| 5382 | REQ-5 | why TCP mappings must survive 124 minutes of idle (compare AWS's 350 seconds) |
| 6888 | §3 | carrier-grade NAT requirements, if you ever have to reason about CGN port budgets |
