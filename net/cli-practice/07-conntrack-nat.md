# `conntrack` and NAT

Conntrack is kernel flow state. Stateful firewall rules and NAT both depend on
it. The command shows what the kernel remembers; it is not a packet capture and
does not prove that an application handled a request.

## Flow model

Conntrack records two directions:

```text
original: client:ephemeral -> server:service
reply:    server:service   -> client:ephemeral
```

With NAT, the addresses or ports visible in these tuples expose the
translation. Netfilter classifies packets as:

- `NEW`: starts a flow or belongs to a flow without both directions confirmed;
- `ESTABLISHED`: belongs to a flow that has seen valid traffic in both
  directions;
- `RELATED`: starts a distinct flow related to an existing one;
- `INVALID`: cannot be associated with valid state;
- `UNTRACKED`: explicitly exempted from tracking.

TCP also has protocol states such as `SYN_SENT`, `SYN_RECV`, `ESTABLISHED`, and
`TIME_WAIT`.

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

In this lab, conntrack state for transit traffic belongs to the router
namespace:

```bash
ip netns exec clp-router conntrack -L
```

Filters are directional. If a filter unexpectedly returns nothing, inspect an
unfiltered entry and distinguish original from reply attributes.

## Reading an entry

A typical TCP entry includes two groups of `src`, `dst`, `sport`, and `dport`.
The first group is the original direction and the second is the reply
direction. It also includes protocol state, timeout, packet/byte counters when
enabled, and marks/status.

For every entry, rewrite it as two arrows on paper. Do not scan the repeated
keys as one flat line.

`[UNREPLIED]` means conntrack has not observed a valid reply. It does not prove
why: the request may not have reached the server, the server may not have
replied, or the reply may have taken another path.

## NAT is stateful

For source NAT:

```text
before router: 10.30.1.10:ephemeral -> 198.51.100.10:8080
after router:  198.51.100.254:mapped -> 198.51.100.10:8080
```

The server replies to the translated source. Conntrack reverses the translation
on return. Only the first packet needs to evaluate the NAT chain to create the
mapping; later packets use stored state.

SNAT changes source after routing, usually at `postrouting`. DNAT changes
destination before routing, usually at `prerouting`. Filtering sees addresses
at particular hook stages, so always state which hook you mean when reasoning
about pre- or post-NAT values.

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

Do not flush all state as a normal troubleshooting step. Deleting state breaks
live flows and can hide the evidence you needed.

## NAT practice

Launch the unsolved topology:

```bash
./scripts/scenarios/nat-missing.sh
```

The server has no route to the private client subnet. Your objective is to let
the client reach server TCP/8080 while making the server see the router's
external address as the peer.

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

Then add a DNAT exercise of your own: publish an internal server port on the
router's external address. Before adding rules, draw packet addresses at
prerouting, routing, forward, and postrouting.

## Capacity awareness

```bash
ip netns exec clp-router sysctl net.netfilter.nf_conntrack_count
ip netns exec clp-router sysctl net.netfilter.nf_conntrack_max
```

When the table is full, new flows can be dropped while established flows
continue. In production, also inspect kernel logs, creation rate, timeouts, and
whether traffic should be tracked at all.

## Retrieval test

Without notes:

- watch TCP flow events;
- list only TCP/8080 entries;
- explain `[UNREPLIED]` without overclaiming;
- draw original/reply tuples for an SNAT flow;
- explain why a NAT rule counter does not increment for every packet;
- compare `nf_conntrack_count` with its limit;
- state why flushing conntrack is risky.
