# Week 08 theory: IPv6 addressing, NDP, SLAAC

Background reading for the IPv6 half of [README08.md](README08.md). Kurose & Ross covers the IPv6
header and transition mechanisms; it barely touches addressing scopes, and it does not cover NDP,
DAD or SLAAC at all — which is unfortunate, because those are the parts you actually operate.

This file contains **no answers** to the exercises. Each section ends with a pointer to the
question it arms you for.

## Where to look for what

One of three files for this week: [NAT](THEORY08-nat.md), **IPv6** (this one),
[SDN](THEORY08-sdn.md).

| README08 item | Section |
| --- | --- |
| Learning goal: read an IPv6 address — `fe80::/10`, `fc00::/7`, `2000::/3` | [Reading an address](#reading-an-address), [The scopes](#the-scopes) |
| Learning goal: understand SLAAC, RA, DAD | [NDP](#ndp-the-protocol-that-replaced-arp-and-more), [DAD](#dad-duplicate-address-detection), [SLAAC](#slaac-end-to-end) |
| Lab step 5: dual-stack, ULA, `radvd`, watch RAs | [SLAAC](#slaac-end-to-end), [Doing this in the lab](#doing-this-in-the-lab) |
| Exercise 1: role of `fe80::/10`, why every interface has one | [Why every interface always has a link-local](#why-every-interface-always-has-a-link-local) |
| Self-check: what is EUI-64 and why not in 2026 | [EUI-64](#eui-64-and-why-not-in-2026) |
| Useful one-liners: `ip -6 addr` / `route` / `neigh` | [Doing this in the lab](#doing-this-in-the-lab) |

Exercise 1's other two bullets and the NAT-related self-checks are in
[THEORY08-nat.md](THEORY08-nat.md); the OpenFlow and Cilium material is in
[THEORY08-sdn.md](THEORY08-sdn.md).

> **One correction to carry into the lab.** README08 step 5 says "add `fd00:1::/64` link-local +
> a ULA". `fd00::/8` is not link-local, it is ULA — the two are different things. Link-local is
> `fe80::/10`, and you never configure it: it appears on its own the moment the interface comes up.
> Read step 5 as "add a ULA prefix and observe the link-local that is already there." Sorting out
> why that is the right reading is most of the learning goal.

---

## Reading an address

128 bits, written as eight 16-bit groups in hex, colon-separated. Two compression rules:

1. Leading zeros within a group may be dropped. `0db8` -> `db8`, `0000` -> `0`.
2. Exactly **one** run of consecutive all-zero groups may be replaced with `::`. Only one, because
   two would be ambiguous — you could not tell how many zeros belonged to each run.

```
2001:0db8:0000:0000:0000:ff00:0042:8329     full
2001:db8:0:0:0:ff00:42:8329                 rule 1
2001:db8::ff00:42:8329                      rule 2
```

RFC 5952 pins down a canonical form so that string comparison works: lowercase hex, always use
`::` on the **longest** zero run (leftmost if tied), and never use `::` for a single zero group
(`2001:db8:0:1:1:1:1:1`, not `2001:db8::1:1:1:1:1`). Linux always prints canonical form, so if
your input does not match `ip -6 addr` output, your input was not canonical.

Prefix notation is the same as IPv4: `2001:db8:acad:1::/64`.

### The structural convention

A global unicast address is conventionally cut like this:

```
|<------- global routing prefix ------->|<- subnet ->|<---- interface ID ---->|
|                 48 bits               |   16 bits  |        64 bits         |
   2001:0db8:acad  :                        0001    :  0000:0000:0000:0001
   ^ from your RIR/ISP                      ^ yours      ^ chosen by the host
```

The /48-to-a-site, /64-per-link split is convention, but the **64-bit interface ID is effectively
mandatory**. SLAAC, RFC 7217 stable-privacy addressing, and solicited-node multicast all assume it.
Subnetting an IPv6 link to /112 to "save addresses" is a classic IPv4 reflex that breaks
autoconfiguration and saves nothing — there is no address scarcity to economise against. Point-to-
point links between routers are the one common exception, where /127 (RFC 6164) is standard
practice specifically to avoid a ping-pong forwarding loop.

*Arms you for:* the learning goal on reading an address.

---

## The scopes

| Prefix | Name | Routable | Notes |
| --- | --- | --- | --- |
| `::/128` | unspecified | no | source address during DAD only. Never a destination |
| `::1/128` | loopback | no | the whole of `127.0.0.0/8` collapsed to one address |
| `fe80::/10` | link-local unicast | link only | auto-configured on every interface, always |
| `fc00::/7` | unique local (ULA) | inside your org | in practice only `fd00::/8` is usable |
| `2000::/3` | global unicast (GUA) | yes | the public internet |
| `ff00::/8` | multicast | scoped, see below | there is no broadcast in IPv6 |
| `2001:db8::/32` | documentation | no | the IPv6 `192.0.2.0/24`. Use it in write-ups |
| `64:ff9b::/96` | NAT64 well-known | — | IPv4 embedded in the low 32 bits |
| `::ffff:0:0/96` | IPv4-mapped | — | internal to dual-stack sockets, not on the wire |

There is no `10.0.0.0/8` equivalent and no NAT in the default IPv6 model. That is the point of the
whole exercise.

### `fe80::/10` — link-local

Allocated as a /10, but in practice **always a /64 with 54 zero bits in between**: RFC 4291
requires bits 11 through 64 to be zero, so every link-local address looks like `fe80::` followed by
a 64-bit interface ID. `fe80::`, `fe81::`, `fe82::`, `feb0::` are all technically inside the /10
and all invalid in practice.

Scope is exactly one link. A router **never forwards** a packet with a link-local source or
destination. Detailed treatment in
[its own section below](#why-every-interface-always-has-a-link-local), since that is exercise 1.

### `fc00::/7` — unique local

RFC 4193. The 8th bit is the L flag:

- `fd00::/8` — L=1, **locally assigned**. This is the half you use.
- `fc00::/8` — L=0, reserved for allocation by some future central registry. That registry was
  never created, so this half is unusable. Anyone typing `fc00::/7` in a config means `fd00::/8`.

The format is prescriptive and the prescription is the interesting part:

```
  fd  |  40-bit global ID  |  16-bit subnet ID  |  64-bit interface ID
  ^ 8    ^ MUST be randomly generated
```

Those 40 bits are required by RFC 4193 §3.2.2 to be **pseudo-randomly** generated, not chosen. The
entire value proposition of ULA is that two organisations who have never met can merge networks
without a collision — which holds only if everybody actually randomises. `fd00:1::/64` (which
README08 suggests, and which everybody writes) has a global ID of all zeros and no such property.
It is fine for a lab; it is exactly the wrong habit for production.

Generate one properly:

```bash
printf 'fd%02x:%04x:%04x::/48\n' $((RANDOM%256)) $((RANDOM)) $((RANDOM))
```

ULA is more controversial than it looks. It is not "IPv6 private addressing" in the RFC 1918 sense,
because the intended IPv6 model is that internal machines carry a GUA and are protected by a
firewall, not by unroutability. Running ULA alongside GUA gives every host two addresses and hands
you RFC 6724 source-address selection problems, including cases where a host prefers IPv4 over a
ULA. Legitimate uses: infrastructure that must keep working across an ISP renumbering, and networks
with no external connectivity at all.

### `2000::/3` — global unicast

Everything from `2000::` to `3fff:ffff:...`. IANA hands regional registries chunks inside it:
`2001::/16`, `2400::/12` (APNIC), `2600::/12` (ARIN), `2800::/12` (LACNIC), `2a00::/12` (RIPE),
`2c00::/12` (AFRINIC). Two historical sub-allocations you will still see referenced and should
recognise as dead: `2002::/16` (6to4, deprecated by RFC 7526) and `3ffe::/16` (the 6bone).

### `ff00::/8` — multicast, and where broadcast went

IPv6 deleted broadcast entirely. Everything that was a broadcast in IPv4 is a multicast group in
IPv6. The second byte carries flags and scope:

```
  ff  |  flags  |  scope  |            group ID
   8      4 bits   4 bits              112 bits
```

Scope values: `1` interface-local, `2` link-local, `5` site-local, `8` organisation, `e` global.
Flag `0` means a permanently-assigned well-known group; `1` (the T bit) means transient.

So `ff02::1` parses as: multicast, well-known, link-local scope, group 1.

Groups you will see in `tcpdump` during the lab:

| Group | Meaning |
| --- | --- |
| `ff02::1` | all-nodes on this link. The nearest thing to a broadcast. RAs go here |
| `ff02::2` | all-routers on this link. RSes go here |
| `ff02::5` / `ff02::6` | OSPFv3 all-SPF-routers / all-DR-routers |
| `ff02::16` | MLDv2 reports |
| `ff02::1:2` | all DHCPv6 relay agents and servers |
| `ff02::1:ff00:0/104` | **solicited-node** — one group per address, see below |

### Solicited-node multicast, worked out

This is the mechanism that makes NDP cheaper than ARP, so do the arithmetic once by hand.

Every node joins, for **each** of its unicast addresses, the group formed by appending that
address's low-order 24 bits to `ff02::1:ff00:0/104`.

Take `2001:db8:1::a2b:3c4d`. Expanded, the last 32 bits are `0a2b:3c4d`. The low **24** bits are
`2b:3c:4d`. So:

```
  prefix:                ff02:0000:0000:0000:0000:0001:ff00:0000   (/104)
  low 24 bits of addr:                                  2b:3c4d
  solicited-node group:  ff02::1:ff2b:3c4d
```

Now map it to Ethernet. An IPv6 multicast address becomes a MAC by prefixing `33:33` to the
address's **last 32 bits**:

```
  ff02::1:ff2b:3c4d   ->  33:33:ff:2b:3c:4d
  ff02::1             ->  33:33:00:00:00:01
```

Here is why that matters. IPv4 ARP is sent to `ff:ff:ff:ff:ff:ff`, so every NIC on the segment
accepts the frame and interrupts its CPU, and every host's IP stack has to parse it and discard it.
An IPv6 neighbor solicitation goes to `33:33:ff:2b:3c:4d`, and a NIC only accepts that frame if
`2b:3c:4d` matches the low 24 bits of one of its own addresses. On a segment with a thousand hosts,
ARP wakes a thousand CPUs and NDP wakes one — the filtering happens in hardware. The 24-bit
truncation means occasional false positives, which is fine; the point is to shed almost all of the
noise, not all of it.

The same 24-bit truncation is also why the interface-ID length matters: two hosts sharing the low
24 bits share a solicited-node group and will both wake for each other's NS.

*Arms you for:* the learning goal on reading an address, and everything in the NDP sections below.

---

## NDP: the protocol that replaced ARP, and more

Neighbor Discovery (RFC 4861) runs over ICMPv6 and absorbs four separate IPv4 mechanisms.

| ICMPv6 type | Message | IPv4 equivalent |
| --- | --- | --- |
| 133 | Router Solicitation (RS) | ICMP router solicitation (rarely used) |
| 134 | Router Advertisement (RA) | DHCP option 3, plus router discovery |
| 135 | Neighbor Solicitation (NS) | ARP request |
| 136 | Neighbor Advertisement (NA) | ARP reply |
| 137 | Redirect | ICMP redirect |

Plus two functions with **no IPv4 equivalent at all**: Duplicate Address Detection, and Neighbor
Unreachability Detection (NUD — the neighbor cache actively probes entries rather than just aging
them out, which is why `ip -6 neigh` shows states like `REACHABLE`, `STALE`, `DELAY`, `PROBE`,
`FAILED`).

Addressing rules for each message, which is what you will be reading off `tcpdump`:

- **RS** — source is the sender's link-local, or `::` if it does not have one yet. Destination
  `ff02::2`.
- **RA** — source **must** be the router's link-local address. Destination `ff02::1` for periodic
  unsolicited ones, or unicast in reply to an RS. A receiver is required to discard an RA whose
  source is not link-local, which is the hard rule behind the whole next section.
- **NS** — destination is the target's solicited-node group for address resolution, or the target's
  unicast address when probing an existing neighbor (NUD).
- **NA** — unicast back to the solicitor, or to `ff02::1` when a node is announcing a change
  unsolicited.

Options carried inside these messages do the real work: Source/Target Link-Layer Address (the
MAC — this is the actual ARP payload equivalent), **Prefix Information** (PIO), MTU, Route
Information, and RDNSS/DNSSL (RFC 8106) which lets an RA carry DNS servers.

A key point people miss: NDP runs entirely over ICMPv6, at layer 3. ARP was a separate EtherType
sitting beside IP. This means NDP inherits IPsec, is media-independent (it works identically over
Ethernet, PPP, tunnels), and — the operational sting — **if you firewall ICMPv6 the way you
firewalled ICMP, you break your own network.** RFC 4890 exists to tell you what you may safely
drop. Filtering NDP is the single most common way to break IPv6.

The hop-limit-255 trick is worth knowing: NDP messages must be sent with hop limit 255 and
discarded if received with anything else. Since a router decrements the hop limit, receiving 255
proves the packet was not forwarded and therefore originated on-link. A cheap and complete defence
against off-link spoofing.

*Arms you for:* the learning goal on SLAAC, RA and DAD.

---

## DAD: duplicate address detection

Before a node assigns **any** unicast address to an interface — link-local, ULA, GUA, autoconfigured
or statically typed in — the address enters the `tentative` state and DAD runs. This has no IPv4
counterpart; IPv4 hosts happily configure duplicate addresses and produce the classic ARP flapping
mess.

The procedure (RFC 4862 §5.4):

1. Join the all-nodes group `ff02::1` and the tentative address's solicited-node group.
2. Send a Neighbor Solicitation with:
   - **source = `::`** (the unspecified address). It has to be — the node has no valid address yet.
     This is the only situation in which `::` appears as a source on the wire.
   - destination = the solicited-node group of the tentative address.
   - target = the tentative address.
   - **no** Source Link-Layer Address option, since there is no valid source address to attach it
     to.
3. Wait `RetransTimer` (default 1000 ms), for `DupAddrDetectTransmits` attempts (default 1).
4. Two ways to fail:
   - an **NA** arrives for that target — somebody already holds the address.
   - an **NS** arrives for that target with source `::` — somebody else is running DAD for the same
     address at the same moment. Both nodes abandon; neither wins.
5. On failure the address is not assigned. If the failed address was the **link-local**, RFC 4862
   requires IPv6 to be disabled on the interface entirely — the interface is unusable, because
   nothing else can proceed. On Linux you will see `dadfailed` in `ip -6 addr`.

Note the sequencing: DAD runs on the link-local address **first**, because every subsequent step
needs a working link-local to send from.

The one-second wait is why IPv6 interfaces sometimes feel slow to come up. RFC 4429 "Optimistic
DAD" lets a host start using an address before DAD completes, accepting a small risk to remove the
delay.

In the lab:

```bash
sudo ip netns exec r tcpdump -ni veth-rin 'icmp6' -vv
# then, in another terminal, flap the interface and watch the tentative NS from ::
sudo ip netns exec r ip link set veth-rin down && sudo ip netns exec r ip link set veth-rin up

sudo ip netns exec r ip -6 addr show tentative
sudo ip netns exec r sysctl net.ipv6.conf.veth-rin.dad_transmits
```

*Arms you for:* the learning goal on DAD.

---

## SLAAC, end to end

Stateless address autoconfiguration (RFC 4862) is how an IPv6 host gets an address with no server
holding any per-client state. Contrast DHCP, where a server owns a lease database.

### The boot sequence

```
 1. interface comes up
 2. form link-local:   fe80::/64  +  interface ID        <- IID from EUI-64 or RFC 7217
 3. DAD on the link-local                                <- NS from ::
 4. join ff02::1, ff02::2 (if a router), solicited-node group
 5. send RS  -->  ff02::2                                <- optional; routers also advertise
                                                            unsolicited every ~200-600s
 6. receive RA  <--  from the router's fe80:: address
 7. for each Prefix Information option with A=1:
        address = advertised prefix (/64)  +  interface ID
 8. DAD on each new address
 9. install a default route via the RA's *source* link-local address
```

A host that sends an RS gets an answer in milliseconds instead of waiting for the next periodic RA.
It retries up to `MAX_RTR_SOLICITATIONS` (3) at 4-second intervals, then gives up and waits.

Note what step 9 says. **The default gateway is the RA's source address, which is a link-local
address.** Not the prefix's `::1`. That is a fact about how IPv6 works, not a configuration choice,
and it is half the answer to exercise 1.

### The RA, field by field

Header fields worth knowing: Cur Hop Limit, the **M** and **O** flags, Router Lifetime (0 means "I
am not a default router, but here is some config anyway"), Reachable Time, Retrans Timer.

The Prefix Information option carries the prefix, its length, two flags, and two lifetimes:

- **A** (autonomous) — use this prefix for SLAAC. Without it, the prefix is advertised but no
  address is formed from it.
- **L** (on-link) — hosts on this link can be reached directly, no router needed.

A and L are independent, and they are independent for a reason. `A=1, L=0` is the standard config
on 3GPP mobile links: form an address from the prefix, but send everything through the router,
because there is no real shared link. Conflating the two is a common misreading of RA output.

The lifetimes are two, not one: **preferred** (address may be used for new connections) and
**valid** (existing connections may keep using it). An address goes preferred -> deprecated ->
invalid, which is what makes graceful renumbering possible — something IPv4 simply cannot do.

### M and O

| M | O | What the host does |
| --- | --- | --- |
| 0 | 0 | Pure SLAAC. Address from the RA. No DHCPv6 at all |
| 0 | 1 | Address from SLAAC; DNS/NTP/domain search from **stateless** DHCPv6 (no lease database) |
| 1 | — | **Stateful** DHCPv6: ask a DHCPv6 server for the address itself |

The historical gap was DNS: SLAAC originally had no way to convey a resolver, which forced O=1 and
a DHCPv6 server just for that. RFC 8106's RDNSS and DNSSL options fixed it by putting DNS servers
directly in the RA. This matters practically because **Android has never implemented DHCPv6** — an
M=1 network simply does not work for Android clients. If you plan an IPv6 network around stateful
DHCPv6, that is the constraint that will find you.

### Prefix delegation is a separate thing

SLAAC configures an *address* on a link. It never delegates a *prefix*. A home router that needs to
number its own internal networks uses DHCPv6-PD (RFC 8415 IA_PD): it asks upstream for, say, a /56,
and then runs `radvd` handing out /64s from it on each internal link.

This is the structural replacement for the IPv4 CPE model. IPv4: one public address, NAT everything
behind it. IPv6: a delegated prefix, every device globally addressable, a firewall instead of a NAT.
Delegating a /64 is a known ISP anti-pattern — it cannot be subnetted, so the customer gets exactly
one link. /56 is the common and sensible assignment.

### Doing this in the lab

`radvd` on the router side, for the ULA in README08 step 5:

```
# /etc/radvd.conf  (run inside netns r)
interface veth-rin {
    AdvSendAdvert on;
    MinRtrAdvInterval 3;
    MaxRtrAdvInterval 10;
    prefix fd00:abcd:1::/64 {
        AdvOnLink on;        # L flag
        AdvAutonomous on;    # A flag
    };
    RDNSS fd00:abcd:1::1 { };
};
```

```bash
sudo ip netns exec r  sysctl -w net.ipv6.conf.all.forwarding=1
sudo ip netns exec r  ip -6 addr add fd00:abcd:1::1/64 dev veth-rin
sudo ip netns exec r  radvd -C /etc/radvd.conf -n -d 5

sudo ip netns exec h1 tcpdump -ni veth-h1 icmp6 -vv     # watch RS then RA then DAD
sudo ip netns exec h1 ip -6 addr                         # the ULA appears with no config
sudo ip netns exec h1 ip -6 route                        # default via fe80::...
sudo ip netns exec h1 ip -6 neigh                        # the NDP cache, with NUD states
```

Things to actually look for in the capture: the RS to `ff02::2`; the RA from a `fe80::` source to
`ff02::1`, with its PIO; the NS from `::` as h1 runs DAD on the new address; and the fact that h1's
default route names a link-local address, not the router's ULA.

Two sysctls that will otherwise waste your afternoon: a host ignores RAs unless
`net.ipv6.conf.<if>.accept_ra` is 1, and a *forwarding* node ignores them unless it is set to 2.

*Arms you for:* the learning goal on SLAAC and RA, and lab step 5.

---

## Why every interface always has a link-local

The cleanest way to answer this is not "because the RFC says so" — it is to ask what would fail
without one.

### The bootstrap problem

Every mechanism that gets you an address needs to send a packet first, and every packet needs a
source address. That is circular unless one address exists before any configuration happens.
Link-local is the base case that terminates the recursion. It is derived purely from local
information — the `fe80::/64` constant plus an interface ID the host computes for itself — so it
requires no server, no router and no link partner.

Concretely, without a link-local address:

- **DAD cannot run**, because DAD is an NDP exchange and NDP needs a source. (DAD itself uses `::`
  precisely because it is the one step that runs before the link-local is valid.)
- **You cannot send a Router Solicitation**, so you cannot ask for a prefix.
- **You cannot receive a usable Router Advertisement**, because an RA must be sourced from a
  link-local address and a receiver must discard one that is not — so you would have no valid way
  to identify the sender as a router on your link.
- **Address resolution stops.** Every NS/NA exchange that turns an address into a MAC needs both
  parties addressable on the link.

So the answer to "why does every interface have one" is that IPv6 was designed to make the link
itself operational before, and independently of, any global configuration.

### The default gateway is a link-local address

This surprises people coming from IPv4, and it is not an accident:

```
$ ip -6 route
fd00:abcd:1::/64 dev eth0  proto kernel  metric 256  pref medium
default via fe80::1 dev eth0  proto ra  metric 1024  pref medium
             ^^^^^^^
```

`fe80::1` here is not a destination. It is a **next hop** — a name that gets resolved through the
neighbor cache into a MAC address, at which point the packet is framed and sent. The packet's IPv6
destination header field stays the remote GUA the whole way. A link-local next hop is perfectly
normal and never appears in any packet's address fields.

Two consequences worth stating:

- **The same `fe80::1` can be configured on every interface of every router in your network.**
  Link-local uniqueness is required per-link, not globally, so there is no conflict. Many operators
  do exactly this so the gateway address is memorable and identical everywhere.
- **The default route survives renumbering.** If your ISP changes your prefix, every GUA on the
  network changes but the link-local next hop does not, so the routing relationship is undisturbed.
  This is a deliberate design property, and it is the same reason OSPFv3 forms adjacencies over
  link-local addresses and BGP can carry link-local next hops (RFC 8950). Routing infrastructure is
  built on top of an address layer that no external event can perturb.

### Not the same as IPv4 169.254/16

IPv4's link-local is a *failure mode*: you get a 169.254 address when DHCP does not answer, and it
usually means something is broken. IPv6's link-local **coexists** with every other address on the
interface at all times, on healthy networks, and carries real infrastructure traffic. Same name,
opposite role.

### Zone identifiers

Because the same link-local address may legitimately exist on many links, a link-local address is
ambiguous without knowing which interface you mean. Hence the zone ID:

```bash
ping6 fe80::1%eth0
ssh   'user@fe80::1%eth0'
```

Two practical traps. In a URI the `%` must itself be percent-encoded:
`http://[fe80::1%25eth0]/`. And `inet_pton()` cannot parse a zone ID at all — you need
`getaddrinfo()`, which is a real source of bugs in code that hand-rolls address parsing.

*Arms you for:* exercise 1, third bullet.

---

## EUI-64, and why not in 2026

### The construction

Modified EUI-64 (RFC 4291 Appendix A) turns a 48-bit MAC into a 64-bit interface ID in two steps.
Take `00:1a:2b:3c:4d:5e`:

```
 1. split after the 24-bit OUI, insert ff:fe

      00:1a:2b : 3c:4d:5e
      00:1a:2b : ff:fe : 3c:4d:5e

 2. flip the U/L bit -- bit 7 of the first octet, i.e. 0x02

      0x00  =  0000 0000
                      ^ flip
      0x02  =  0000 0010

 result IID:      021a:2bff:fe3c:4d5e
 link-local:      fe80::21a:2bff:fe3c:4d5e
 with a ULA:      fd00:abcd:1::21a:2bff:fe3c:4d5e
```

The `ff:fe` in the middle of an interface ID is the giveaway — spot it in `ip -6 addr` output and
you know that host is using EUI-64 and you can read its MAC straight off the address.

Why the bit flip? In IEEE 802 the U/L bit means 0 = universally administered, 1 = locally
administered. IPv6 inverted the sense so that a hand-written short address like `2001:db8::1` reads
as "locally administered" rather than colliding with the universal space. It is a convenience for
humans typing addresses, and it is the detail everyone gets backwards.

### The problems

1. **It is trivially reversible.** Strip `ff:fe`, unflip the bit, and you have the MAC. That
   exposes the OUI, so the vendor and often the device model, to anyone who sees a packet.
2. **It is constant across networks.** Your laptop's IID is the same at home, at work and in a
   café. Only the prefix changes. A server that sees you from three networks can trivially link the
   three sessions to one physical device. This is a supercookie you cannot clear, sitting in the
   network layer where no browser setting reaches it.
3. **It defeats ISP prefix rotation.** Some ISPs rotate a subscriber's prefix daily as a privacy
   measure. With EUI-64 the IID is unchanged across the rotation, so a passive observer can stitch
   the old prefix to the new one by following the constant suffix. APNIC published measurements
   doing exactly this at ISP scale.
4. **It shrinks the scan space.** The theoretical defence against address scanning is that a /64
   holds 2^64 addresses. If addresses are EUI-64, an attacker who guesses the vendor OUI is
   scanning a far smaller space than 2^64, and it collapses further for a homogeneous fleet
   deployed from one batch.
5. **One EUI-64 device de-anonymises everything behind it.** This is the argument worth citing. A
   measurement study across a 15-million-subscriber ISP found that a single always-on EUI-64
   device — usually an IoT appliance that never got privacy extensions — acts as a stable anchor
   for the whole prefix, defeating the privacy extensions of every *other* device on that home
   network. 19% of subscribers were affected. Your own hygiene is not sufficient.

### What replaced it

- **RFC 8981 temporary addresses** (obsoleting RFC 4941, the "privacy extensions"): a random IID,
  regenerated periodically, used as the source for outbound connections. Coexists with a stable
  address so inbound service still works. Solves problem 2 for outbound traffic and nothing else.
- **RFC 7217 stable-privacy addresses**: `IID = F(prefix, interface, network_id, dad_counter,
  secret_key)`. Three properties in one construction — no scannable pattern, a *different* address
  on every network you join, and a *stable* address for as long as you stay on one network. That
  last property is what temporary addresses give up, and it is why 7217 is the right default: your
  firewall rules and AAAA records keep working.
- **RFC 8064** changed the recommended default for SLAAC from EUI-64 to stable-privacy. That is
  the citation for "why not in 2026" — it is not a preference, it is the current standards-track
  recommendation, and it has been since 2017.

On Linux:

```bash
sysctl net.ipv6.conf.eth0.addr_gen_mode    # 0 = EUI-64, 2 = stable-privacy, 3 = random
sysctl net.ipv6.conf.eth0.use_tempaddr     # 0 = off, 1 = on, 2 = on and prefer temporary
ip -6 addr show dev eth0                   # temporary addresses are flagged "temporary"
```

NetworkManager has defaulted to `ipv6.addr-gen-mode=stable-privacy` for years, and typically also
randomises the MAC itself, which addresses the same problem one layer down. If you see `ff:fe` in
the middle of an address on a modern desktop distribution, someone has explicitly configured it —
which happens most often on servers, routers and containers, where the defaults are older.

The last thing worth saying in the answer: none of this makes EUI-64 wrong on a link where
tracking is not a threat — a point-to-point router link, a lab netns. It is a bad *default*, not a
banned mechanism.

*Arms you for:* the self-check on EUI-64.

---

## Reading list

All verified reachable. Where a piece is long, the section to actually read is named.

**Address anatomy and scopes**

- [IPv6 address types (networkacademy.io)](https://www.networkacademy.io/ccna/ipv6/ipv6-address-types)
  — the single best free page for this topic. Walks the whole space (GUA and its aggregation, ULA,
  link-local, loopback, unspecified, IPv4-embedded), then multicast with a table of well-known
  groups and a full derivation of solicited-node from `ff02::1:ff00:0/104`.
- [Back to basics: the IPv6 address types (Infoblox)](https://www.infoblox.com/blog/ipv6-coe/back-to-basics-the-ipv6-address-types/)
  — short and opinionated, notable for arguing *why* ULA is usually not the right answer and why
  GUA is the default even internally. Read it before deciding you want ULA everywhere.

**NDP and DAD**

- [IPv6 Neighbor Discovery Protocol (networkacademy.io)](https://www.networkacademy.io/ccna/ipv6/neighbor-discovery-protocol)
  — packet-by-packet through all five message types, including the derivation of the
  `33:33:xx:xx:xx:xx` multicast MAC and why NDP beats ARP broadcast. Read this one first.
- [IPv6 Neighbor Discovery (Cisco IOS XE configuration guide)](https://www.cisco.com/c/en/us/td/docs/ios-xml/ios/ipv6_basic/configuration/xe-3se/3850/ip6-neighb-disc-xe.html)
  — vendor-authoritative and precise on DAD (tentative state, unspecified source, collision
  handling) and on NUD, which almost nothing else covers properly. Read the DAD and NUD sections.

**SLAAC**

- [IPv6 SLAAC (networkacademy.io)](https://www.networkacademy.io/ccna/ipv6/stateless-address-autoconfiguration-slaac)
  — follows a host through all five steps with Wireshark screenshots, then covers M/O and router
  preference. The comment thread is unusually good on RDNSS as an alternative to O=1.
- [Introducing DHCPv6 prefix delegation (LACNIC)](https://blog.lacnic.net/en/introducing-dhcpv6-prefix-delegation/)
  — Tom Coffeen explains PD by direct contrast with the IPv4 CPE + NAT model, and why /64
  delegation is broken and /56 became standard.

**Link-local**

- [What's the deal with IPv6 link-local addresses? (APNIC)](https://blog.apnic.net/2020/03/30/whats-the-deal-with-ipv6-link-local-addresses/)
  — the best single article for exercise 1. Strong on the contrast with IPv4 link-local, and the
  clearest treatment anywhere of zone identifiers, `inet_pton` limitations and `%25` in URIs.
- [FE80::1 is a perfectly valid IPv6 default gateway address (Infoblox)](https://www.infoblox.com/blog/ipv6-coe/fe80-1-is-a-perfectly-valid-ipv6-default-gateway-address/)
  — answers the "what breaks without one" question directly, and explains why a link-local next
  hop in a routing table is normal and why the same `fe80::1` can be reused everywhere.

**EUI-64 and privacy**

- [Defeating IPv6 prefix rotation privacy (APNIC)](https://blog.apnic.net/2022/01/31/defeating-ipv6-prefix-rotation-privacy/)
  — the most precise short statement of the EUI-64 construction anywhere, followed by the
  real-world consequence: tracking subscribers across daily prefix rotations.
- [How IoT devices can endanger your IPv6 privacy (APNIC)](https://blog.apnic.net/2022/06/10/iot-devices-endanger-ipv6-privacy/)
  — the strongest single argument against EUI-64, with measurement across 14.4M devices. This is
  the one to cite in the self-check.
- [A brief history of recent advances in IPv6 security, part I: addressing (APNIC)](https://blog.apnic.net/2020/08/24/a-brief-history-of-recent-advances-in-ipv6-security-part-i-addressing/)
  — by Fernando Gont, author of RFC 7217. Explains the `Hash(Prefix, Secret)` design and its three
  properties, and narrates why temporary addresses alone were insufficient.

## RFCs, lookup only

| RFC | Look at | For |
| --- | --- | --- |
| 4291 | §2.4, §2.5.6, §2.7 | the address type table, link-local format, multicast scopes |
| 4291 | Appendix A | the modified EUI-64 construction, including the U/L bit flip |
| 5952 | §4 | canonical text representation, if two tools disagree on a string |
| 4193 | §3.1-3.2 | ULA format and the requirement that the global ID be random |
| 4861 | §4.1-4.6 | the five NDP message formats and their options |
| 4861 | §7.3 | NUD and the neighbor cache state machine |
| 4862 | §5.4 | DAD, in full. Short and worth reading if you read one RFC section this week |
| 4862 | §5.5 | how a host processes a Prefix Information option |
| 8200 | §3, §4 | the base header and extension header chain — the K&R reading, in the original |
| 8106 | §5 | RDNSS and DNSSL option formats |
| 8064 | §1, §3 | the recommendation to stop defaulting to EUI-64 |
| 7217 | §5 | the stable-privacy IID generation algorithm |
| 4890 | §4.4 | which ICMPv6 types you must not filter, before you write a v6 firewall |
