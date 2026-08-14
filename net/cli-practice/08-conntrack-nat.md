# `conntrack` and NAT

Conntrack is kernel **flow state**. Stateful firewall rules and NAT both
depend on it. The `conntrack` command shows what the kernel remembers. It
is not a packet capture and does not prove that an application handled a
request.

## Where you are

You can write a stateful forward policy in both iptables and nft, and you
know SNAT attaches after routing while DNAT attaches before. This module
is the table those features read and write, plus a topology where NAT is
the only correct fix.

## What you need, and what you do not

Need: four-tuples from [`ss`](03-ss.md); capture points from
[`tcpdump`](04-tcpdump.md); NAT hook placement from
[`iptables`](06-iptables.md) and [`nft`](07-nftables.md).

Do not need: helper modules for FTP, SIP, or other "related" protocols.
`RELATED` exists; this lab's HTTP and iperf3 do not require helpers.

## Why the kernel must remember

Without memory, a router that allowed a SYN to `10.20.2.10:8080` cannot
recognize the SYN-ACK to `10.20.1.10:45678` as "the other half of that
conversation." Conntrack stores both halves.

It also stores translations. After SNAT, the server sees the router's
address, not the client's. Replies come back to the router. Only conntrack
knows which internal client should receive them.

## Flow model

Conntrack records two directions:

```text
original: client:ephemeral → server:service
reply:    server:service   → client:ephemeral
```

With NAT, the addresses or ports visible in these tuples expose the
translation. Draw two arrows on paper for every entry. Do not scan the
repeated `src=` keys as one flat line.

Netfilter classifies packets as:

- `NEW`: starts a flow, or belongs to a flow without both directions
  confirmed;
- `ESTABLISHED`: belongs to a flow that has seen valid traffic in both
  directions;
- `RELATED`: starts a distinct flow related to an existing one;
- `INVALID`: cannot be associated with valid state;
- `UNTRACKED`: explicitly exempted from tracking.

These are **conntrack** classes. TCP also has protocol states such as
`SYN_SENT`, `SYN_RECV`, `ESTABLISHED`, and `TIME_WAIT`. A packet can be
conntrack-`ESTABLISHED` while you are looking at a TCP `TIME_WAIT` entry
that is winding down. Read both fields; do not merge them.

`[UNREPLIED]` means conntrack has not observed a valid reply. It does not
prove why: the request may not have reached the server, the server may not
have replied, or the reply may have taken another path. Capture to
distinguish those.

## Command anatomy

```text
conntrack OPERATION [filters]

-L list       -E event stream       -D delete matching
-F flush      -C count              -S statistics
```

Observation:

```bash
conntrack -L
conntrack -L -p tcp
conntrack -L -p tcp --dport 8080
conntrack -E -p tcp
conntrack -C
conntrack -S
```

In this lab, conntrack state for **transit** traffic belongs to the router
namespace:

```bash
ip netns exec clp-router conntrack -L
```

The endpoints have their own TCP sockets (`ss`). The router has the
forwarding flow (`conntrack`). Those are different objects.

Filters are directional. If a filter unexpectedly returns nothing, inspect
an unfiltered entry and distinguish original from reply attributes.
`--dport 8080` matches the original destination port in typical client →
server flows; after some NAT it may not be the port you think.

## NAT is stateful

The NAT incident uses a different picture on purpose:

```text
clp-client                clp-router                   clp-server
10.30.1.10/24      10.30.1.1 | 198.51.100.254     198.51.100.10/24
    c-eth0 -------- r-inside | r-outside -------- s-eth0
         private 10.30.1.0/24 | "public" 198.51.100.0/24
```

The server has **no route** to `10.30.1.0/24`. That is realistic: a public
host does not know your living-room LAN. If the client sends with source
`10.30.1.10`, the server's reply is unroutable. The fix is **not** to add
that route on the server (the incident forbids it). The fix is SNAT:

```text
before router: 10.30.1.10:ephemeral → 198.51.100.10:8080
after router:  198.51.100.254:mapped → 198.51.100.10:8080
```

The server replies to `198.51.100.254`, which it **can** reach. Conntrack
reverses the translation on return. Only the first packet needs to evaluate
the NAT chain to create the mapping; later packets use stored state.

SNAT changes source after routing, usually at `postrouting`. DNAT changes
destination before routing, usually at `prerouting`. Filtering sees
addresses at particular hook stages, so always state which hook you mean
when reasoning about pre- or post-NAT values.

Worked DNAT (port publish), for the exercise at the end:

```text
outside SYN:  client → 198.51.100.254:8080
after DNAT:   client → 10.30.1.10:8080     (route now finds the inside)
```

Without DNAT, the router would treat `198.51.100.254:8080` as **INPUT** to
itself, not as FORWARD to the client namespace.

## Capacity awareness

```bash
ip netns exec clp-router sysctl net.netfilter.nf_conntrack_count
ip netns exec clp-router sysctl net.netfilter.nf_conntrack_max
```

When the table is full, new flows can be dropped while established flows
continue. That matches the signature "existing connections work, new ones
fail." In production, also inspect kernel logs, creation rate, timeouts,
and whether traffic should be tracked at all.

Do not flush all state as a normal troubleshooting step. `conntrack -F`
breaks live flows and can hide the evidence you needed.

### Possible gap: helpers, SNAT port exhaustion, hairpin NAT

FTP-style protocols open extra ports; conntrack **helpers** associate
them. Many public IPs plus many clients can exhaust SNAT port mappings.
Two internal hosts reaching each other via the NAT's public address
(**hairpin**) need extra rules. None of these are required to pass
incident 4. If you meet them later, you already know where to look:
conntrack tuples and the postrouting/prerouting hooks.

## Muscle-memory drill

```bash
./scripts/setup-routed.sh
./scripts/setup-services.sh
```

Open an event stream in one shell:

```bash
ip netns exec clp-router conntrack -E -p tcp
```

Generate HTTP and iperf3 traffic from another shell. For each:

1. identify `NEW`, `UPDATE`, and `DESTROY` events;
2. draw original and reply tuples;
3. correlate them with `ss` on both endpoints;
4. correlate them with a `tcpdump` capture on each router interface;
5. count and narrowly filter current entries.

## NAT practice

Launch the unsolved topology:

```bash
./scripts/scenarios/nat-missing.sh
```

The server has no route to the private client subnet. Your objective is to
let the client reach server TCP/8080 while making the server see the
router's external address as the peer.

Do this twice:

1. implement source NAT with `iptables`;
2. reset the scenario and implement the equivalent with `nft`.

For each implementation, prove all four:

- the client receives an HTTP response;
- a capture on the inside shows the private source;
- a capture on the outside shows the translated source;
- conntrack shows enough tuple information to reverse the reply.

Run the non-revealing validator:

```bash
./checks/check-nat.sh
```

Then add a DNAT exercise of your own: publish an internal server port on
the router's external address. Before adding rules, draw packet addresses
at prerouting, routing, forward, and postrouting.

## Retrieval test

Without notes:

- watch TCP flow events;
- list only TCP/8080 entries;
- explain `[UNREPLIED]` without overclaiming;
- draw original/reply tuples for an SNAT flow;
- explain why a NAT rule counter does not increment for every packet;
- compare `nf_conntrack_count` with its limit;
- state why flushing conntrack is risky.

Next: [integrated troubleshooting](09-troubleshooting.md).
